package com.google.ai.edge.gallery.api

import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.cio.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.coroutines.flow.collect
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import java.io.File

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
    private var engine: Engine? = null
    private var currentModelPath: String? = null
    private val lock = Any()

    fun getOrCreateEngine(
        modelPath: String,
        nativeLibDir: String,
        useNpu: Boolean = true,
        cacheDir: String? = null,
    ): Engine {
        synchronized(lock) {
            if (engine != null && currentModelPath == modelPath) {
                return engine!!
            }
            engine?.close()
            engine = null

            val backend = if (useNpu) {
                try {
                    Backend.NPU(nativeLibraryDir = nativeLibDir)
                } catch (e: Exception) {
                    Log.w(TAG, "NPU unavailable, falling back to GPU: ${e.message}")
                    Backend.GPU()
                }
            } else {
                Backend.GPU()
            }

            val config = EngineConfig(
                modelPath = modelPath,
                backend = backend,
                cacheDir = cacheDir,
            )

            val newEngine = Engine(config)
            newEngine.initialize()
            engine = newEngine
            currentModelPath = modelPath
            Log.i(TAG, "Engine initialized: $modelPath backend=$backend")
            return newEngine
        }
    }

    fun close() {
        synchronized(lock) {
            engine?.close()
            engine = null
            currentModelPath = null
        }
    }

    fun currentModelId(): String = currentModelPath?.let {
        File(it).nameWithoutExtension
    } ?: "no-model-loaded"
}

// ── Ktor 서버 ────────────────────────────────────────────────────

class OpenAiApiServer(
    private val port: Int = 8080,
    private val nativeLibDir: String,
    private val cacheDir: String,
    /** 첫 번째 모델 경로 (null이면 요청 시 지정 필요) */
    private val defaultModelPath: String? = null,
) {
    private var server: ApplicationEngine? = null

    fun start() {
        server = embeddedServer(CIO, port = port) {
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

        // 모델 경로 결정: X-Model-Path 헤더 > defaultModelPath
        val modelPath = call.request.headers["X-Model-Path"]
            ?: defaultModelPath
            ?: run {
                call.respond(
                    HttpStatusCode.BadRequest,
                    ErrorResponse(
                        ErrorDetail(
                            "No model loaded. Set X-Model-Path header or configure defaultModelPath."
                        )
                    )
                )
                return
            }

        if (!File(modelPath).exists()) {
            call.respond(
                HttpStatusCode.BadRequest,
                ErrorResponse(ErrorDetail("Model file not found: $modelPath"))
            )
            return
        }

        val engine = try {
            EngineManager.getOrCreateEngine(
                modelPath = modelPath,
                nativeLibDir = nativeLibDir,
                cacheDir = cacheDir,
            )
        } catch (e: Exception) {
            Log.e(TAG, "Engine init failed", e)
            call.respond(
                HttpStatusCode.InternalServerError,
                ErrorResponse(ErrorDetail("Engine initialization failed: ${e.message}"))
            )
            return
        }

        // Conversation 생성
        val responseText = StringBuilder()
        try {
            val systemMsg = req.messages.firstOrNull { it.role == "system" }?.content
            val convConfig = ConversationConfig(
                systemInstruction = systemMsg?.let { Contents.of(it) },
                samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 1.0),
            )
            engine.createConversation(convConfig).use { conversation ->
                val userMessages = req.messages.filter { it.role != "system" }

                // 이전 assistant 메시지들을 히스토리로 추가 (있으면)
                // LiteRT-LM은 대화 히스토리를 sendMessageAsync로 순서대로 보내면 됨
                for (i in userMessages.indices) {
                    val msg = userMessages[i]
                    if (i < userMessages.size - 1) {
                        // 히스토리 메시지 (마지막 제외) → 상태만 쌓기
                        if (msg.role == "user") {
                            conversation.sendMessageAsync(msg.content).collect { /* discard */ }
                        }
                    } else {
                        // 마지막 user 메시지 → 실제 응답 생성
                        if (msg.role == "user") {
                            conversation.sendMessageAsync(msg.content).collect { token ->
                                responseText.append(token)
                            }
                        }
                    }
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
