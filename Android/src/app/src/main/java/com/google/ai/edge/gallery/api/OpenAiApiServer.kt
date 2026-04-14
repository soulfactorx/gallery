package com.google.ai.edge.gallery.api

import android.util.Log
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
        call.respond(ModelListResponse(data = listOf(ModelInfo(id = EngineManager.currentModelId()))))
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

        val conversation = EngineManager.getConversation() ?: run {
            call.respond(
                HttpStatusCode.ServiceUnavailable,
                ErrorResponse(ErrorDetail("No model loaded. Please open chat in the app first."))
            )
            return
        }

        val nonSystemMessages = req.messages.filter { it.role != "system" }
        val lastMsg = nonSystemMessages.lastOrNull { it.role == "user" }?.content ?: ""
        val chatId = "chatcmpl-${System.currentTimeMillis()}"
        val modelId = EngineManager.currentModelId()

        val responseText = StringBuilder()
        try {
            conversation.sendMessageAsync(lastMsg).collect { token: Any ->
                responseText.append(token.toString())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            call.respond(
                HttpStatusCode.InternalServerError,
                ErrorResponse(ErrorDetail("Inference failed: ${e.message}"))
            )
            return
        }

        call.respond(
            ChatCompletionResponse(
                id = chatId,
                created = System.currentTimeMillis() / 1000,
                model = modelId,
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
        )
    }
}
