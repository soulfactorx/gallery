package com.google.ai.edge.gallery.api

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.google.ai.edge.gallery.MainActivity  // Gallery의 MainActivity

private const val TAG = "ApiServerService"
private const val CHANNEL_ID = "gemma_api_server"
private const val NOTIFICATION_ID = 9001

class ApiServerService : Service() {

    private var apiServer: OpenAiApiServer? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification("Starting..."))
        Log.i(TAG, "Service created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val port = intent.getIntExtra(EXTRA_PORT, 8080)
                val modelPath = intent.getStringExtra(EXTRA_MODEL_PATH)
                startServer(port, modelPath)
            }
            ACTION_STOP -> stopSelf()
        }
        return START_STICKY
    }

    override fun onDestroy() {
        apiServer?.stop()
        apiServer = null
        Log.i(TAG, "Service destroyed")
        super.onDestroy()
    }

    private fun startServer(port: Int, defaultModelPath: String?) {
        apiServer?.stop()

        apiServer = OpenAiApiServer(
            port = port,
            nativeLibDir = applicationInfo.nativeLibraryDir,
            cacheDir = cacheDir.absolutePath,
            defaultModelPath = defaultModelPath,
        )

        try {
            apiServer!!.start()
            updateNotification("Running on port $port ✓")
            Log.i(TAG, "Server started on port $port")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start server", e)
            updateNotification("Failed to start: ${e.message}")
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Gemma API Server",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "OpenAI-compatible API server running on-device"
        }
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(channel)
    }

    private fun buildNotification(status: String): Notification {
        val openAppIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stopIntent = PendingIntent.getService(
            this, 1,
            Intent(this, ApiServerService::class.java).apply { action = ACTION_STOP },
            PendingIntent.FLAG_IMMUTABLE,
        )

        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Gemma NPU API")
            .setContentText(status)
            .setSmallIcon(android.R.drawable.ic_menu_share)
            .setContentIntent(openAppIntent)
            .setOngoing(true)
            .addAction(
                Notification.Action.Builder(
                    null, "Stop", stopIntent
                ).build()
            )
            .build()
    }

    private fun updateNotification(status: String) {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, buildNotification(status))
    }

    companion object {
        const val ACTION_START = "com.google.ai.edge.gallery.api.ACTION_START"
        const val ACTION_STOP = "com.google.ai.edge.gallery.api.ACTION_STOP"
        const val EXTRA_PORT = "port"
        const val EXTRA_MODEL_PATH = "model_path"

        /** Gallery의 어느 Activity에서든 호출 가능한 헬퍼 */
        fun start(context: Context, port: Int = 8080, modelPath: String? = null) {
            val intent = Intent(context, ApiServerService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_PORT, port)
                modelPath?.let { putExtra(EXTRA_MODEL_PATH, it) }
            }
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, ApiServerService::class.java).apply {
                action = ACTION_STOP
            }
            context.startService(intent)
        }
    }
}
