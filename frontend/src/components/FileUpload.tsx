'use client'

import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { Upload, FileText, X, Loader2, CheckCircle, AlertCircle, BarChart3, Database } from 'lucide-react'

interface FileUploadProps {
  onUpload?: (files: FileList) => void
  className?: string
}

interface UploadFile extends File {
  uploadId: string
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error'
  progress: number
  taskId?: string
  stats?: {
    entities_extracted?: number
    relationships_created?: number
    error_count?: number
  }
  error?: string
}

interface ProcessingStats {
  task_id: string
  status: string
  progress: number
  current_file: string
  files_processed: number
  total_files: number
  entities_extracted: number
  errors: string[]
  eta_seconds: number
}

export default function FileUpload({ onUpload, className }: FileUploadProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [files, setFiles] = useState<UploadFile[]>([])
  const [sessionId, setSessionId] = useState<string>()
  const [processingStats, setProcessingStats] = useState<ProcessingStats | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const progressIntervalRef = useRef<ReturnType<typeof setInterval>>()

  const supportedTypes = [
    'text/plain',
    'text/csv',
    'application/json',
    'application/pdf',
    'text/markdown'
  ]

  const maxFiles = 20
  const maxFileSize = 50 * 1024 * 1024 // 50MB
  const maxTotalSize = 200 * 1024 * 1024 // 200MB

  const validateFiles = (fileList: FileList): UploadFile[] => {
    const validFiles: UploadFile[] = []
    const totalSize = Array.from(fileList).reduce((sum, file) => sum + file.size, 0)

    if (fileList.length > maxFiles) {
      alert(`Too many files. Maximum ${maxFiles} files allowed.`)
      return []
    }

    if (totalSize > maxTotalSize) {
      alert(`Total file size too large. Maximum ${maxTotalSize / (1024 * 1024)}MB allowed.`)
      return []
    }

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i] as UploadFile

      if (!supportedTypes.includes(file.type)) {
        alert(`Unsupported file type: ${file.name}. Please upload supported formats.`)
        continue
      }

      if (file.size > maxFileSize) {
        alert(`File too large: ${file.name}. Maximum ${maxFileSize / (1024 * 1024)}MB per file.`)
        continue
      }

      file.uploadId = `upload-${Date.now()}-${i}`
      file.status = 'pending'
      file.progress = 0
      validFiles.push(file)
    }

    return validFiles
  }

  const handleFilesSelected = useCallback((fileList: FileList) => {
    const validFiles = validateFiles(fileList)
    if (validFiles.length > 0) {
      setFiles(validFiles)
      onUpload?.(fileList)
    }
  }, [onUpload])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const droppedFiles = e.dataTransfer.files
    if (droppedFiles.length > 0) {
      handleFilesSelected(droppedFiles)
    }
  }, [handleFilesSelected])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesSelected(e.target.files)
    }
  }

  const uploadFiles = async () => {
    if (files.length === 0) return

    // Reset all files to pending status
    setFiles(prev => prev.map(file => ({ ...file, status: 'pending' as const, progress: 0 })))

    try {
      // Create FormData for multipart upload
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      // Upload files to get session ID
      const uploadResponse = await fetch('http://localhost:8000/api/upload/files', {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed: ${uploadResponse.statusText}`)
      }

      const uploadData = await uploadResponse.json()
      const uploadSessionId = uploadData.session_id
      setSessionId(uploadSessionId)

      // Update file statuses to uploaded
      setFiles(prev => prev.map(file => ({
        ...file,
        status: 'uploading' as const,
        progress: 100
      })))

      // Start processing
      const processResponse = await fetch('http://localhost:8000/api/upload/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: uploadSessionId,
          entity_types: ['Person', 'Organization', 'Technology', 'Concept']
        })
      })

      if (!processResponse.ok) {
        throw new Error(`Processing start failed: ${processResponse.statusText}`)
      }

      const processData = await processResponse.json()
      const taskId = processData.task_id

      // Mark as processing and start progress monitoring
      setFiles(prev => prev.map(file => ({ ...file, status: 'processing' as const })))

      // Set up polling for progress updates
      progressIntervalRef.current = setInterval(() => checkProgress(taskId), 1000)

    } catch (error: any) {
      console.error('Upload error:', error)
      setFiles(prev => prev.map(file => ({
        ...file,
        status: 'error' as const,
        error: error.message
      })))
    }
  }

  const checkProgress = async (taskId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/upload/progress/${taskId}`)
      if (!response.ok) {
        console.error('Failed to get progress')
        return
      }

      const stats: ProcessingStats = await response.json()
      setProcessingStats(stats)

      // Update file statuses based on progress
      setFiles(prev => prev.map(file => {
        if (file.name === stats.current_file) {
          return { ...file, status: 'processing' as const, taskId }
        }
        return file
      }))

      // Stop polling when complete
      if (stats.status === 'completed' || stats.status === 'failed') {
        if (progressIntervalRef.current) {
          clearInterval(progressIntervalRef.current)
          progressIntervalRef.current = undefined
        }

        // Update all files with completion status and stats
        setFiles(prev => prev.map(file => ({
          ...file,
          status: stats.status === 'completed' ? 'completed' as const : 'error' as const,
          stats: {
            entities_extracted: stats.entities_extracted,
            relationships_created: stats.relationships_created,
            error_count: stats.errors.length
          }
        })))
      }

    } catch (error) {
      console.error('Progress check error:', error)
    }
  }

  const removeFile = (uploadId: string) => {
    setFiles(prev => prev.filter(file => file.uploadId !== uploadId))
  }

  const clearAll = () => {
    setFiles([])
    setSessionId(undefined)
    setProcessingStats(null)
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current)
      progressIntervalRef.current = undefined
    }
  }

  const getFileIcon = (file: UploadFile) => {
    if (file.type === 'application/pdf') return <FileText className="h-5 w-5" />
    if (file.type === 'text/csv') return <Database className="h-5 w-5" />
    if (file.type === 'application/json') return <FileText className="h-5 w-5" />
    return <FileText className="h-5 w-5" />
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-gray-500'
      case 'uploading': return 'text-blue-500'
      case 'processing': return 'text-yellow-500'
      case 'completed': return 'text-green-500'
      case 'error': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Upload className="h-4 w-4" />
      case 'uploading':
      case 'processing': return <Loader2 className="h-4 w-4 animate-spin" />
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'error': return <AlertCircle className="h-4 w-4" />
      default: return <Upload className="h-4 w-4" />
    }
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(true)}
        title="Upload documents for processing"
        className={`${className} btn-hover-lift focus-glow`}
      >
        <Upload className="h-4 w-4 mr-2" />
        Upload Files
      </Button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Document Processing System
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            {/* Upload Zone */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Upload Documents</CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    isDragOver ? 'border-primary bg-primary/5' : 'border-gray-300'
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".txt,.csv,.json,.pdf,.md"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                  <Upload className={`h-12 w-12 mx-auto mb-4 ${isDragOver ? 'text-primary' : 'text-gray-400'}`} />
                  <p className="text-lg font-medium mb-2">
                    {isDragOver ? 'Drop files here' : 'Drag & drop files here'}
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    Or click to select files (CSV, TXT, JSON, PDF, Markdown)
                  </p>
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose Files
                  </Button>
                </div>

                <div className="text-xs text-muted-foreground mt-2">
                  Max {maxFiles} files • Max {maxFileSize / (1024 * 1024)}MB per file • Max {maxTotalSize / (1024 * 1024 * 1024)}GB total
                </div>
              </CardContent>
            </Card>

            {/* File List */}
            {files.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm flex items-center justify-between">
                    Uploaded Files ({files.length})
                    <Button variant="outline" size="sm" onClick={clearAll}>
                      Clear All
                    </Button>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {files.map((file) => (
                      <div key={file.uploadId} className="flex items-center gap-3 p-3 border rounded-lg">
                        <div className={`p-2 rounded ${getStatusColor(file.status)} bg-current/10`}>
                          {getStatusIcon(file.status)}
                        </div>

                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            {getFileIcon(file)}
                            <span className="font-medium truncate">{file.name}</span>
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {(file.size / 1024 / 1024).toFixed(1)}MB • {file.status}
                            {file.stats && (
                              <span className="ml-2 text-green-600">
                                • {file.stats.entities_extracted || 0} entities
                              </span>
                            )}
                          </div>
                          {processingStats && processingStats.current_file === file.name && (
                            <div className="mt-1">
                              <Progress value={processingStats.progress} className="h-1" />
                              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                                <span>Processing...</span>
                                <span>{processingStats.entities_extracted} entities</span>
                              </div>
                            </div>
                          )}
                          {file.error && (
                            <div className="text-sm text-red-600 mt-1">
                              {file.error}
                            </div>
                          )}
                        </div>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(file.uploadId)}
                          disabled={file.status === 'processing'}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>

                  {files.every(f => f.status === 'completed') && (
                    <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2 text-green-800">
                        <CheckCircle className="h-5 w-5" />
                        <span className="font-medium">All files processed successfully!</span>
                      </div>
                      <div className="mt-2 grid grid-cols-3 gap-4 text-sm">
                        <div className="flex items-center gap-1">
                          <FileText className="h-4 w-4" />
                          <span>{files.length} files</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <BarChart3 className="h-4 w-4" />
                          <span>{files.reduce((sum, f) => sum + (f.stats?.entities_extracted || 0), 0)} entities</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Database className="h-4 w-4" />
                          <span>{files.reduce((sum, f) => sum + (f.stats?.relationships_created || 0), 0)} relationships</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {files.some(f => f.status === 'pending') && (
                    <div className="mt-4 flex gap-2">
                      <Button onClick={uploadFiles} className="flex-1">
                        Process Files ({files.filter(f => f.status === 'pending').length})
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
