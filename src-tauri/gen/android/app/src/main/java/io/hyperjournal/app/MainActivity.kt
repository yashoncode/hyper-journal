package io.hyperjournal.app

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import androidx.activity.enableEdgeToEdge

class MainActivity : TauriActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    enableEdgeToEdge()
    super.onCreate(savedInstanceState)
    askForTheDocumentsFolder()
  }

  /**
   * The journal lives in the shared Documents folder, so that it is still there
   * after this app is gone. On Android 11 and later that folder is only
   * reachable under All-files access, which cannot be granted from a normal
   * permission dialog -- it is a settings screen the person has to visit.
   *
   * Camera and microphone are not asked for here. The page requests them
   * through getUserMedia, and wry's WebChromeClient already turns that into the
   * runtime prompt at the moment they are actually needed, which is the better
   * time to ask.
   */
  private fun askForTheDocumentsFolder() {
    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.R) return
    if (Environment.isExternalStorageManager()) return
    runCatching {
      startActivity(
        Intent(
          Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
          Uri.parse("package:$packageName")
        )
      )
    }.onFailure {
      // some builds do not ship that screen; the app falls back to its own
      // storage rather than failing to start
      runCatching { startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
    }
  }
}
