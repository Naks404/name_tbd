package com.example.myapplication

import android.app.Application
import androidx.camera.core.ImageProxy
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

class FieldAssistantApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
    }
}

object YuvConverter {
    fun imageProxyToNv21(image: ImageProxy): ByteArray {
        val width = image.width
        val height = image.height
        val ySize = width * height
        val uvSize = width * height / 4
        val nv21 = ByteArray(ySize + uvSize * 2)

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val yRowStride = image.planes[0].rowStride
        val yPixelStride = image.planes[0].pixelStride

        val uRowStride = image.planes[1].rowStride
        val uPixelStride = image.planes[1].pixelStride

        val vRowStride = image.planes[2].rowStride
        val vPixelStride = image.planes[2].pixelStride

        var pos = 0

        if (yRowStride == width) {
            yBuffer.position(0)
            yBuffer.get(nv21, 0, ySize)
            pos += ySize
        } else {
            var yBufferPos = 0
            for (row in 0 until height) {
                yBuffer.position(yBufferPos)
                yBuffer.get(nv21, pos, width)
                pos += width
                yBufferPos += yRowStride
            }
        }

        if (vPixelStride == 2 && vRowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            val vSize = vBuffer.remaining()
            vBuffer.get(nv21, pos, minOf(vSize, nv21.size - pos))
        } else {
            for (row in 0 until height / 2) {
                var vBufferPos = row * vRowStride
                var uBufferPos = row * uRowStride
                for (col in 0 until width / 2) {
                    nv21[pos++] = vBuffer.get(vBufferPos)
                    nv21[pos++] = uBuffer.get(uBufferPos)
                    vBufferPos += vPixelStride
                    uBufferPos += uPixelStride
                }
            }
        }
        
        yBuffer.rewind()
        uBuffer.rewind()
        vBuffer.rewind()

        return nv21
    }
}

