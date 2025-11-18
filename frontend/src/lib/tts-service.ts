interface TTSModel {
  name: string
  description: string
  language?: string
}

interface GenerateSpeechParams {
  text: string
  model: string
  outputPath: string
}

export class TTSService {
  private serverUrl = process.env.COQUI_TTS_SERVER_URL || 'http://localhost:5002'
  private publicAudioDir = 'public/audio'

  async checkServerHealth(): Promise<boolean> {
    try {
      const response = await fetch(`${this.serverUrl}/api/tts`, {
        method: 'HEAD'
      })
      return response.ok
    } catch (error) {
      console.warn('TTS server health check failed:', error)
      return false
    }
  }

  async listAvailableModels(): Promise<TTSModel[]> {
    try {
      const response = await fetch(`${this.serverUrl}/api/tts`)
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`)
      }

      const data = await response.json()
      const models = data.models || []

      return models.map((model: unknown) => {
        const modelObj = model as Record<string, unknown>
        return {
          name: (modelObj.name as string) || String(model),
          description: (modelObj.description as string) || (modelObj.name as string) || 'TTS Model',
          language: (modelObj.language as string) || 'en'
        }
      })
    } catch (error) {
      console.warn('Failed to fetch TTS models:', error)
      // Return fallback models
      return [
        {
          name: 'browser-tts',
          description: 'Browser TTS (Default)',
          language: 'en'
        }
      ]
    }
  }

  async generateSpeech({ text, model, outputPath }: GenerateSpeechParams): Promise<string> {
    try {
      const params = new URLSearchParams({
        text,
        model_name: model,
        vocoder_name: 'vocoder_models/en/ljspeech/hifigan_v2',
        use_cuda: 'false'
      })

      const response = await fetch(`${this.serverUrl}/api/tts?${params}`, {
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error(`TTS server responded with status: ${response.status}`)
      }

      // Get the audio data as blob
      const audioBlob = await response.blob()

      // Generate filename if not provided
      const filename = outputPath || `tts_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.wav`

      // For now, we'll just return the filename since we can't actually write files in the browser
      // The server should handle file storage
      console.log(`TTS audio generated: ${filename}`)
      return filename

    } catch (error) {
      console.error('TTS generation error:', error)
      throw new Error(`Failed to generate speech: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  // Helper method to get the public URL for an audio file
  getAudioUrl(filename: string): string {
    return `/audio/${filename}`
  }
}
