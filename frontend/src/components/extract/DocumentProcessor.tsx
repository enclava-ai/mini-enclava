"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText, X, AlertCircle, CheckCircle2, Loader2, DollarSign, Clock } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { extractApi } from "@/lib/api-client"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface Template {
  id: string
  description: string
}

interface ProcessResult {
  success: boolean
  job_id: string
  data: Record<string, any>
  validation_errors: string[]
  validation_warnings: string[]
  processing_time_ms: number
  tokens_used: number
  cost_cents: number
}

export function DocumentProcessor() {
  const [templates, setTemplates] = useState<Template[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string>("")
  const [file, setFile] = useState<File | null>(null)
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<ProcessResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { toast } = useToast()

  useEffect(() => {
    loadTemplates()
  }, [])

  const loadTemplates = async () => {
    try {
      const response = await extractApi.listTemplates()
      setTemplates(response.templates || [])
      if (response.templates?.length > 0 && !selectedTemplate) {
        setSelectedTemplate(response.templates[0].id)
      }
    } catch (err) {
      toast({
        title: "Error",
        description: "Failed to load templates",
        variant: "destructive",
      })
    }
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    if (selectedFile) {
      setFile(selectedFile)
      setResult(null)
      setError(null)
    }
  }

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return
    setFile(files[0])
    setResult(null)
    setError(null)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    handleFileSelect(e.dataTransfer.files)
  }

  const processDocument = async () => {
    if (!file) {
      setError("Please select a file")
      return
    }

    setProcessing(true)
    setError(null)
    setResult(null)

    try {
      const response = await extractApi.processDocument(file, selectedTemplate)
      setResult(response)

      if (response.success) {
        toast({
          title: "Success",
          description: "Document processed successfully",
        })
      } else {
        toast({
          title: "Warning",
          description: "Document processed with validation errors",
          variant: "destructive",
        })
      }
    } catch (err: any) {
      const errorMessage = err?.details?.detail || err?.message || "Processing failed"
      setError(errorMessage)
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      })
    } finally {
      setProcessing(false)
    }
  }

  const clearForm = () => {
    setFile(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return "N/A"
    if (typeof value === "object") {
      return JSON.stringify(value, null, 2)
    }
    return String(value)
  }

  const formatCost = (cents: number): string => {
    return `$${(cents / 100).toFixed(4)}`
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Upload Document</CardTitle>
          <CardDescription>
            Upload invoices, receipts, or other documents for automated data extraction
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Template Selection */}
          <div className="space-y-2">
            <Label htmlFor="template">Template</Label>
            <Select value={selectedTemplate} onValueChange={setSelectedTemplate} disabled={processing}>
              <SelectTrigger>
                <SelectValue placeholder="Select a template" />
              </SelectTrigger>
              <SelectContent>
                {templates.map((template) => (
                  <SelectItem key={template.id} value={template.id}>
                    {template.id} - {template.description}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* File Upload */}
          <div className="space-y-2">
            <Label>Select Image or PDF</Label>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragOver
                  ? 'border-primary bg-primary/5'
                  : 'border-gray-300 hover:border-gray-400'
              } cursor-pointer`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
              <div className="space-y-2">
                <p className="text-lg font-medium">
                  {dragOver ? 'Drop file here' : 'Drop file here or click to browse'}
                </p>
                <p className="text-sm text-gray-500">
                  Supported: JPG, PNG, PDF
                </p>
                <p className="text-xs text-gray-400">
                  Maximum file size: 10MB
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.pdf"
                onChange={handleFileChange}
                className="hidden"
                disabled={processing}
              />
            </div>
            {file && (
              <p className="text-sm text-muted-foreground">
                Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <Button
              onClick={processDocument}
              disabled={!file || processing}
              className="flex-1"
            >
              {processing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                'Process Document'
              )}
            </Button>
            <Button
              variant="outline"
              onClick={clearForm}
              disabled={processing}
            >
              Clear
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Display */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Extracted Data</span>
              <div className="flex gap-2">
                <Badge variant="outline" className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {result.processing_time_ms}ms
                </Badge>
                <Badge variant="outline" className="flex items-center gap-1">
                  <DollarSign className="h-3 w-3" />
                  {formatCost(result.cost_cents)}
                </Badge>
                <Badge variant="outline">
                  {result.tokens_used} tokens
                </Badge>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {result.success && result.data ? (
              <>
                {/* Extracted Fields */}
                <div className="grid gap-3">
                  {Object.entries(result.data).map(([key, value]) => (
                    <div key={key} className="grid grid-cols-[180px_1fr] gap-4 p-3 bg-muted/50 rounded-lg">
                      <span className="font-semibold text-sm">{key}:</span>
                      <div className="text-sm">
                        {typeof value === "object" && value !== null ? (
                          <pre className="bg-background border rounded p-2 text-xs overflow-x-auto">
                            {formatValue(value)}
                          </pre>
                        ) : (
                          formatValue(value)
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Validation Errors */}
                {result.validation_errors && result.validation_errors.length > 0 && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold mb-2">Validation Errors</div>
                      <ul className="list-disc list-inside space-y-1">
                        {result.validation_errors.map((error, idx) => (
                          <li key={idx}>{error}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Validation Warnings */}
                {result.validation_warnings && result.validation_warnings.length > 0 && (
                  <Alert>
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold mb-2">Validation Warnings</div>
                      <ul className="list-disc list-inside space-y-1">
                        {result.validation_warnings.map((warning, idx) => (
                          <li key={idx}>{warning}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {/* Raw JSON */}
                <details className="cursor-pointer">
                  <summary className="text-sm font-medium text-primary hover:underline">
                    View Raw JSON
                  </summary>
                  <pre className="mt-2 bg-muted p-4 rounded-lg text-xs overflow-x-auto">
                    {JSON.stringify(result.data, null, 2)}
                  </pre>
                </details>
              </>
            ) : (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Processing completed but no structured data extracted.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
