import fs from 'fs'
import path from 'path'

export interface SystemPrompt {
  id: string
  name: string
  content: string
  createdAt: string
  updatedAt: string
}

const PROMPTS_FILE_PATH = path.join(process.cwd(), 'data', 'system-prompts.json')

function readPromptsFromFile(): SystemPrompt[] {
  try {
    if (!fs.existsSync(PROMPTS_FILE_PATH)) {
      // Create default prompts if file doesn't exist
      const defaultPrompts: SystemPrompt[] = [{
        id: 'default',
        name: 'Default System Prompt',
        content: 'Be concise',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }]
      fs.writeFileSync(PROMPTS_FILE_PATH, JSON.stringify(defaultPrompts, null, 2))
      return defaultPrompts
    }

    const content = fs.readFileSync(PROMPTS_FILE_PATH, 'utf8')
    return JSON.parse(content)
  } catch (error) {
    console.error('Error reading system prompts:', error)
    return []
  }
}

function writePromptsToFile(prompts: SystemPrompt[]): void {
  try {
    fs.writeFileSync(PROMPTS_FILE_PATH, JSON.stringify(prompts, null, 2))
  } catch (error) {
    console.error('Error writing system prompts:', error)
    throw error
  }
}

export function readPrompts(): SystemPrompt[] {
  return readPromptsFromFile()
}

export function getPromptById(id: string): SystemPrompt | undefined {
  const prompts = readPromptsFromFile()
  return prompts.find(prompt => prompt.id === id)
}

export function createPrompt(name: string, content: string): SystemPrompt {
  const prompts = readPromptsFromFile()
  const newPrompt: SystemPrompt = {
    id: `prompt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    name,
    content,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString()
  }

  prompts.push(newPrompt)
  writePromptsToFile(prompts)
  return newPrompt
}

export function updatePrompt(id: string, name: string, content: string): SystemPrompt | null {
  const prompts = readPromptsFromFile()
  const promptIndex = prompts.findIndex(prompt => prompt.id === id)

  if (promptIndex === -1) {
    return null
  }

  prompts[promptIndex] = {
    ...prompts[promptIndex],
    name,
    content,
    updatedAt: new Date().toISOString()
  }

  writePromptsToFile(prompts)
  return prompts[promptIndex]
}

export function deletePrompt(id: string): boolean {
  const prompts = readPromptsFromFile()
  const filteredPrompts = prompts.filter(prompt => prompt.id !== id)

  if (filteredPrompts.length === prompts.length) {
    return false // No prompt found to delete
  }

  writePromptsToFile(filteredPrompts)
  return true
}
