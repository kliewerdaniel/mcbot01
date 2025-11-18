import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

// Types
export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export interface Session {
  id: string
  title: string
  messages: Message[]
  model: string
  createdAt: Date
  updatedAt: Date
}

export interface SystemPrompt {
  id: string
  name: string
  content: string
  createdAt: string
  updatedAt: string
}

interface OllamaModel {
  name: string
  model: string
  modified_at: string
  size: number
  digest: string
  details: {
    parent_model: string
    format: string
    family: string
    families: string[]
    parameter_size: string
    quantization_level: string
  }
}

interface ChatStore {
  // State
  messages: Message[]
  input: string
  isLoading: boolean
  models: { name: string; size?: string }[]
  selectedModel: string
  selectedPromptId: string
  inputRows: number
  isNearBottom: boolean
  isUserScrolling: boolean
  sessions: Session[]
  currentSessionId: string | null
  systemPrompts: SystemPrompt[]

  // Actions
  setInput: (input: string) => void
  setIsLoading: (loading: boolean) => void
  setSelectedModel: (model: string) => void
  setSelectedPromptId: (promptId: string) => void
  setInputRows: (rows: number) => void
  setIsNearBottom: (near: boolean) => void
  setIsUserScrolling: (scrolling: boolean) => void
  addMessage: (message: Message) => void
  updateLastMessage: (content: string) => void
  loadSessions: () => void
  saveSession: () => void
  createNewSession: () => void
  clearCurrentChat: () => void
  switchSession: (sessionId: string) => void
  renameSession: (sessionId: string, newTitle: string) => void
  deleteSession: (sessionId: string) => void
  clearAllSessions: () => void
  loadModels: () => void
  fetchSystemPrompts: () => void
}

interface UIRelatedStore {
  // State
  isEditorOpen: boolean
  isHistoryOpen: boolean
  editingSessionId: string | null
  editingTitle: string
  voices: SpeechSynthesisVoice[]
  selectedVoice: string
  playingMessageId: string | null
  voiceEnhancement: boolean
  autoPlayEnabled: boolean
  ttsModels: { name: string; description: string; language?: string }[]
  selectedTTSModel: string
  isTTSLoading: boolean
  ttsLoadingMessages: string[]

  // Actions
  setIsEditorOpen: (open: boolean) => void
  setIsHistoryOpen: (open: boolean) => void
  setEditingSessionId: (id: string | null) => void
  setEditingTitle: (title: string) => void
  loadVoices: () => void
  setSelectedVoice: (voice: string) => void
  setPlayingMessageId: (id: string | null) => void
  setVoiceEnhancement: (enabled: boolean) => void
  setAutoPlayEnabled: (enabled: boolean) => void
  loadTTSModels: () => void
  setSelectedTTSModel: (model: string) => void
  setTTSLoading: (loading: boolean) => void
  addTTSLoadingMessage: (messageId: string) => void
  removeTTSLoadingMessage: (messageId: string) => void
  isMessageTTSLoading: (messageId: string) => boolean
}

interface PromptStore {
  // State
  prompts: SystemPrompt[]
  isLoading: boolean

  // Actions
  loadPrompts: () => void
  addPrompt: (prompt: Omit<SystemPrompt, 'id' | 'createdAt' | 'updatedAt'>) => Promise<void>
  updatePrompt: (id: string, updates: Partial<Omit<SystemPrompt, 'id' | 'createdAt' | 'updatedAt'>>) => Promise<void>
  deletePrompt: (id: string) => Promise<void>
}

// Chat Store
export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      // Initial state
      messages: [],
      input: '',
      isLoading: false,
      models: [],
      selectedModel: 'mistral-small3.2:latest',
      selectedPromptId: 'default',
      inputRows: 3,
      isNearBottom: true,
      isUserScrolling: false,
      sessions: [],
      currentSessionId: null,
      systemPrompts: [],

      // Actions
      setInput: (input) => set({ input }),
      setIsLoading: (loading) => set({ isLoading: loading }),
      setSelectedModel: (model) => set({ selectedModel: model }),
      setSelectedPromptId: (promptId) => set({ selectedPromptId: promptId }),
      setInputRows: (rows) => set({ inputRows: rows }),
      setIsNearBottom: (near) => set({ isNearBottom: near }),
      setIsUserScrolling: (scrolling) => set({ isUserScrolling: scrolling }),

      addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
      })),

      updateLastMessage: (content) => set((state) => ({
        messages: state.messages.map((msg, index) =>
          index === state.messages.length - 1
            ? { ...msg, content: msg.content + content }
            : msg
        )
      })),

      loadSessions: () => {
        try {
          const stored = localStorage.getItem('chat-sessions')
          if (stored) {
            const rawSessions = JSON.parse(stored)
            const sessions: Session[] = rawSessions.map((session: Record<string, unknown>) => ({
              ...session,
              createdAt: new Date(session.createdAt as string),
              updatedAt: new Date(session.updatedAt as string)
            })) as Session[]
            set({ sessions })
          }
        } catch (error) {
          console.error('Error loading sessions:', error)
        }
      },

      saveSession: () => {
        const state = get()
        if (!state.currentSessionId || state.messages.length === 0) return

        const updatedSessions = state.sessions.map(session =>
          session.id === state.currentSessionId
            ? {
                ...session,
                messages: state.messages,
                model: state.selectedModel,
                updatedAt: new Date()
              }
            : session
        )

        localStorage.setItem('chat-sessions', JSON.stringify(updatedSessions))
        set({ sessions: updatedSessions })
      },

      createNewSession: () => {
        const newSessionId = `session-${Date.now()}`
        const newSession: Session = {
          id: newSessionId,
          title: `Chat ${get().sessions.length + 1}`,
          messages: [],
          model: get().selectedModel,
          createdAt: new Date(),
          updatedAt: new Date()
        }

        const updatedSessions = [...get().sessions, newSession]
        localStorage.setItem('chat-sessions', JSON.stringify(updatedSessions))

        set({
          sessions: updatedSessions,
          currentSessionId: newSessionId,
          messages: [],
          selectedModel: get().selectedModel
        })
      },

      clearCurrentChat: () => {
        set({ messages: [] })
      },

      switchSession: (sessionId) => {
        const session = get().sessions.find(s => s.id === sessionId)
        if (session) {
          set({
            currentSessionId: sessionId,
            messages: session.messages,
            selectedModel: session.model
          })
        }
      },

      renameSession: (sessionId, newTitle) => {
        const updatedSessions = get().sessions.map(session =>
          session.id === sessionId
            ? { ...session, title: newTitle }
            : session
        )
        localStorage.setItem('chat-sessions', JSON.stringify(updatedSessions))
        set({ sessions: updatedSessions })
      },

      deleteSession: (sessionId) => {
        const updatedSessions = get().sessions.filter(session => session.id !== sessionId)
        localStorage.setItem('chat-sessions', JSON.stringify(updatedSessions))
        set({ sessions: updatedSessions })

        // Clear current chat if the deleted session was active
        if (get().currentSessionId === sessionId) {
          set({ currentSessionId: null, messages: [] })
        }
      },

      clearAllSessions: () => {
        localStorage.removeItem('chat-sessions')
        set({ sessions: [], currentSessionId: null, messages: [] })
      },

      loadModels: async () => {
        try {
          const response = await fetch('/api/models')
          if (response.ok) {
            const data = await response.json()
            if (data.models && Array.isArray(data.models)) {
              const models = data.models.map((model: OllamaModel) => ({
                name: model.name,
                size: model.size ? `${(model.size / (1024 * 1024 * 1024)).toFixed(1)}GB` : undefined
              }))
              set({ models })
            } else {
              console.error('Unexpected models API response format:', data)
              // Fallback to empty array
              set({ models: [] })
            }
          } else {
            console.error('Failed to load models:', response.statusText)
            // Fallback to empty array
            set({ models: [] })
          }
        } catch (error) {
          console.error('Error loading models:', error)
          // Fallback to empty array
          set({ models: [] })
        }
      },

      fetchSystemPrompts: async () => {
        try {
          const response = await fetch('/api/system-prompts')
          if (response.ok) {
            const data = await response.json()
            set({ systemPrompts: data.prompts })
          } else {
            console.error('Error fetching system prompts:', response.statusText)
          }
        } catch (error) {
          console.error('Error fetching system prompts:', error)
        }
      }
    }),
    {
      name: 'chat-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        selectedModel: state.selectedModel,
        selectedPromptId: state.selectedPromptId
      })
    }
  )
)

// UI Store
export const useUIStore = create<UIRelatedStore>()(
  persist(
    (set, get) => ({
      // Initial state
      isEditorOpen: false,
      isHistoryOpen: false,
      editingSessionId: null,
      editingTitle: '',
      voices: [],
      selectedVoice: '',
      playingMessageId: null,
      voiceEnhancement: false,
      autoPlayEnabled: false,
      ttsModels: [{ name: 'browser-tts', description: 'Browser TTS (Default)', language: 'en' }],
      selectedTTSModel: 'browser-tts',
      isTTSLoading: false,
      ttsLoadingMessages: [],

      // Actions
      setIsEditorOpen: (open) => set({ isEditorOpen: open }),
      setIsHistoryOpen: (open) => set({ isHistoryOpen: open }),
      setEditingSessionId: (id) => set({ editingSessionId: id }),
      setEditingTitle: (title) => set({ editingTitle: title }),

      loadVoices: () => {
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          const voices = speechSynthesis.getVoices()
          set({ voices })

          // Set default voice if not already set
          if (!get().selectedVoice && voices.length > 0) {
            const englishVoice = voices.find(voice => voice.lang.startsWith('en'))
            set({ selectedVoice: (englishVoice || voices[0]).voiceURI })
          }
        }
      },

      setSelectedVoice: (voice) => set({ selectedVoice: voice }),
      setPlayingMessageId: (id) => set({ playingMessageId: id }),
      setVoiceEnhancement: (enabled) => set({ voiceEnhancement: enabled }),
      setAutoPlayEnabled: (enabled) => set({ autoPlayEnabled: enabled }),

      loadTTSModels: async () => {
        try {
          const response = await fetch('/api/tts')
          const data = await response.json()
          if (data.success) {
            set({ ttsModels: data.models })
          }
        } catch (error) {
          console.warn('Failed to load TTS models:', error)
        }
      },

      setSelectedTTSModel: (model) => set({ selectedTTSModel: model }),
      setTTSLoading: (loading) => set({ isTTSLoading: loading }),

      addTTSLoadingMessage: (messageId) => set((state) => ({
        ttsLoadingMessages: [...state.ttsLoadingMessages, messageId]
      })),

      removeTTSLoadingMessage: (messageId) => set((state) => ({
        ttsLoadingMessages: state.ttsLoadingMessages.filter(id => id !== messageId)
      })),

      isMessageTTSLoading: (messageId) => get().ttsLoadingMessages.includes(messageId)
    }),
    {
      name: 'ui-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        selectedVoice: state.selectedVoice,
        voiceEnhancement: state.voiceEnhancement,
        autoPlayEnabled: state.autoPlayEnabled,
        selectedTTSModel: state.selectedTTSModel
      })
    }
  )
)

// Prompt Store
export const usePromptStore = create<PromptStore>()((set, get) => ({
  // Initial state
  prompts: [],
  isLoading: false,

  // Actions
  loadPrompts: async () => {
    set({ isLoading: true })
    try {
      const response = await fetch('/api/system-prompts')
      if (!response.ok) {
        throw new Error('Failed to fetch prompts')
      }
      const data = await response.json()
      set({ prompts: data.prompts, isLoading: false })
    } catch (error) {
      console.error('Error loading prompts:', error)
      set({ isLoading: false })
      throw error
    }
  },

  addPrompt: async (promptData) => {
    try {
      const response = await fetch('/api/system-prompts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: promptData.name,
          content: promptData.content
        })
      })

      if (!response.ok) {
        throw new Error('Failed to create prompt')
      }

      const data = await response.json()
      set((state) => ({
        prompts: [...state.prompts, data.prompt]
      }))
    } catch (error) {
      console.error('Error adding prompt:', error)
      throw error
    }
  },

  updatePrompt: async (id, updates) => {
    try {
      if (!updates.name || !updates.content) {
        throw new Error('Name and content are required')
      }

      const response = await fetch(`/api/system-prompts/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: updates.name,
          content: updates.content
        })
      })

      if (!response.ok) {
        throw new Error('Failed to update prompt')
      }

      const data = await response.json()
      set((state) => ({
        prompts: state.prompts.map(prompt =>
          prompt.id === id ? data.prompt : prompt
        )
      }))
    } catch (error) {
      console.error('Error updating prompt:', error)
      throw error
    }
  },

  deletePrompt: async (id) => {
    try {
      const response = await fetch(`/api/system-prompts/${id}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        throw new Error('Failed to delete prompt')
      }

      set((state) => ({
        prompts: state.prompts.filter(prompt => prompt.id !== id)
      }))
    } catch (error) {
      console.error('Error deleting prompt:', error)
      throw error
    }
  }
}))
