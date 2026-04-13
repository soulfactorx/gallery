package com.google.ai.edge.gallery.api

import android.util.Log
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.cio.CIO
import io.ktor.server.engine.ApplicationEngine
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

private const val TAG = "OpenAiApiServer"

// ── OpenAI 호환 데이터 모델 ──────────────────────────────────────

@Serializable
data class ChatMessage(
    val role: String,
    val content: String,
)

@Serializable
data class ChatCompletionRequest(
    val model: String = "gemma",
    val messages: List<ChatMessage>,
    val temperature: Float = 1.0f,
    val max_tokens: Int = 2048,
    val stream: Boolean = false,
)

@Serializable
data class Choice(
    val index: Int,
    val message: ChatMessage,
    val finish_reason: String,
)

@Serializable
data class Usage(
    val prompt_tokens: Int,
    val completion_tokens: Int,
    val total_tokens: Int,
)

@Serializable
data class ChatCompletionResponse(
    val id: String,
    val `object`: String = "chat.completion",
    val created: Long,
    val model: String,
    val choices: List<Choice>,
    val usage: Usage,
)

@Serializable
data class ModelInfo(
    val id: String,
    val `object`: String = "model",
    val owned_by: String = "local",
)

@Serializable
data class ModelListResponse(
    val `object`: String = "list",
    val data: List<ModelInfo>,
)

@Serializable
data class ErrorResponse(
    val error: ErrorDetail,
)

@Serializable
data class ErrorDetail(
    val message: String,
    val type: String = "invalid_request_error",
)

// ── 엔진 관리 ────────────────────────────────────────────────────

object EngineManager {
    var sharedEngine: Engine? = null
    var sharedConversation: com.google.ai.edge.litertlm.Conversation? = null
    var sharedModelId: String = "no-model-loaded"

    fun getEngine(): Engine? = sharedEngine
    fun getConversation(): com.google.ai.edge.litertlm.Conversation? = sharedConversation
    fun currentModelId(): String = sharedModelId

    fun close() {
        sharedEngine = null
        sharedConversation = null
        sharedModelId = "no-model-loaded"
    }
}

// ── Ktor 서버 ────────────────────────────────────────────────────

class OpenAiApiServer(
    private val port: Int = 8080,
    private val nativeLibDir: String,
    private val cacheDir: String,
    private val defaultModelPath: String? = null,
) {
    private var server: ApplicationEngine? = null

    fun start() {
        server = embeddedServer(factory = CIO, port = port) {
            install(ContentNegotiation) {
                json(Json {
                    prettyPrint = false
                    isLenient = true
                    ignoreUnknownKeys = true
                })
            }
            install(CORS) {
                anyHost()
                allowHeader(HttpHeaders.ContentType)
                allowHeader(HttpHeaders.Authorization)
                allowMethod(HttpMethod.Options)
                allowMethod(HttpMethod.Post)
                allowMethod(HttpMethod.Get)
            }
            routing {
                get("/") {
                    call.respondText("Gemma NPU API Server running on port $port")
                }
                get("/v1/models") { handleModels(call) }
                post("/v1/chat/completions") { handleChatCompletions(call) }
            }
        }.start(wait = false)
        Log.i(TAG, "API server started on port $port")
    }

    fun stop() {
        server?.stop(1000, 2000)
        EngineManager.close()
        Log.i(TAG, "API server stopped")
    }

    // GET /v1/models
    private suspend fun handleModels(call: ApplicationCall) {
        val models = listOf(
            ModelInfo(id = EngineManager.currentModelId()),
        )
        call.respond(ModelListResponse(data = models))
    }

    // POST /v1/chat/completions
    private suspend fun handleChatCompletions(call: ApplicationCall) {
        val req = try {
            call.receive<ChatCompletionRequest>()
        } catch (e: Exception) {
            call.respond(
                HttpStatusCode.BadRequest,
                ErrorResponse(ErrorDetail("Invalid request body: ${e.message}"))
            )
            return
        }

        val engine = EngineManager.getEngine() ?: run {
            call.respond(
                HttpStatusCode.ServiceUnavailable,
                ErrorResponse(ErrorDetail("No model loaded. Please load a model in the app first."))
            )
            return
        }

        val responseText = StringBuilder()
        try {
            // 앱의 conversation 닫기
            val appConversation = EngineManager.sharedConversation
            if (appConversation != null) {
                try { appConversation.cancelProcess() } catch (e: Exception) { }
                delay(300)
                try { appConversation.close() } catch (e: Exception) { }
                delay(300)
                EngineManager.sharedConversation = null
            }

            // system 프롬프트 추출
            val systemMsg = req.messages.firstOrNull { it.role == "system" }?.content

            // RisuAI 설정으로 새 conversation 생성
            val apiConversation = engine.createConversation(
                ConversationConfig(
                    systemInstruction = systemMsg?.let { Contents.of(it) },
                    samplerConfig = null,
                )
            )

            try {
                // 대화 히스토리 재현 (마지막 메시지 제외)
                val nonSystemMessages = req.messages.filter { it.role != "system" }
                for (i in 0 until nonSystemMessages.size - 1) {
                    val msg = nonSystemMessages[i]
                    if (msg.role == "user") {
                        apiConversation.sendMessageAsync(msg.content).collect { }
                    }
                }
                // 마지막 user 메시지로 실제 응답 생성
                val lastMsg = nonSystemMessages.lastOrNull { it.role == "user" }?.content ?: ""
                apiConversation.sendMessageAsync(lastMsg).collect { token ->
                    responseText.append(token)
                }
            } finally {
                // API conversation 닫고 앱 conversation 복원
                try { apiConversation.close() } catch (e: Exception) { }
                delay(300)
                try {
                    val newAppConversation = engine.createConversation(
                        ConversationConfig(samplerConfig = null)
                    )
                    EngineManager.sharedConversation = newAppConversation
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to restore app conversation", e)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            call.respond(
                HttpStatusCode.InternalServerError,
                ErrorResponse(ErrorDetail("Inference failed: ${e.message}"))
            )
            return
        }

        val response = ChatCompletionResponse(
            id = "chatcmpl-${System.currentTimeMillis()}",
            created = System.currentTimeMillis() / 1000,
            model = EngineManager.currentModelId(),
            choices = listOf(
                Choice(
                    index = 0,
                    message = ChatMessage(
                        role = "assistant",
                        content = responseText.toString(),
                    ),
                    finish_reason = "stop",
                )
            ),
            usage = Usage(
                prompt_tokens = req.messages.sumOf { it.content.length / 4 },
                completion_tokens = responseText.length / 4,
                total_tokens = (req.messages.sumOf { it.content.length } + responseText.length) / 4,
            ),
        )
        call.respond(response)
    }
}
