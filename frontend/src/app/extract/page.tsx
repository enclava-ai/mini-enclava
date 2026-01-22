"use client"

import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { ExtractManager } from '@/components/extract/ExtractManager'

export default function ExtractPage() {
  return (
    <ProtectedRoute>
      <ExtractPageContent />
    </ProtectedRoute>
  )
}

function ExtractPageContent() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Document Extraction</h1>
        <p className="text-muted-foreground mt-2">
          Extract structured data from invoices, receipts, and documents using vision language models
        </p>
      </div>
      <ExtractManager />
    </div>
  )
}
