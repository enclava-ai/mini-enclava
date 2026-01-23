"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { Copy, Check, Code2, Terminal, FileCode } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function IntegrationGuide() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const baseUrl = typeof window !== 'undefined' ? window.location.origin : 'https://your-domain.com'

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  const CopyButton = ({ text, id }: { text: string; id: string }) => (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => copyToClipboard(text, id)}
      className="absolute top-2 right-2"
    >
      {copiedCode === id ? (
        <Check className="h-4 w-4" />
      ) : (
        <Copy className="h-4 w-4" />
      )}
    </Button>
  )

  const CodeBlock = ({ code, language, id }: { code: string; language: string; id: string }) => (
    <div className="relative">
      <pre className="bg-muted p-4 rounded-lg overflow-x-auto">
        <code className={`language-${language} text-sm`}>{code}</code>
      </pre>
      <CopyButton text={code} id={id} />
    </div>
  )

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <CardTitle>API Integration</CardTitle>
          <CardDescription>
            Use the Extract API to process documents programmatically. Create an API key with extract permissions to get started.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="font-semibold mb-2">Base URL</h3>
            <code className="bg-muted px-3 py-1.5 rounded text-sm">{baseUrl}/api/v1/extract</code>
          </div>

          <Alert>
            <AlertDescription>
              <strong>Authentication:</strong> Include your API key in the Authorization header: <code className="bg-muted px-2 py-0.5 rounded text-xs">Bearer YOUR_API_KEY</code>
            </AlertDescription>
          </Alert>

          <div>
            <h3 className="font-semibold mb-2">Required Scope</h3>
            <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
              <li><code className="bg-muted px-2 py-0.5 rounded">extract</code> - Access to all Extract operations</li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Process Document */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            Process Document
          </CardTitle>
          <CardDescription>
            POST /api/v1/extract/process - Extract structured data from a document
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="curl" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="curl">cURL</TabsTrigger>
              <TabsTrigger value="python">Python</TabsTrigger>
              <TabsTrigger value="typescript">TypeScript</TabsTrigger>
            </TabsList>

            <TabsContent value="curl" className="mt-4">
              <CodeBlock
                id="process-curl"
                language="bash"
                code={`curl -X POST "${baseUrl}/api/v1/extract/process" \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -F "file=@invoice.pdf" \\
  -F "template=detailed_invoice" \\
  -F 'context={"company_name":"Acme Corp","currency":"USD"}'`}
              />
            </TabsContent>

            <TabsContent value="python" className="mt-4">
              <CodeBlock
                id="process-python"
                language="python"
                code={`import requests

url = "${baseUrl}/api/v1/extract/process"
headers = {
    "Authorization": "Bearer YOUR_API_KEY"
}

# Open file in binary mode
with open("invoice.pdf", "rb") as file:
    files = {"file": file}
    data = {
        "template": "detailed_invoice",
        "context": '{"company_name":"Acme Corp","currency":"USD"}'
    }

    response = requests.post(url, headers=headers, files=files, data=data)
    result = response.json()

    if result["success"]:
        print(f"Job ID: {result['job_id']}")
        print(f"Extracted data: {result['data']}")
    else:
        print(f"Error: {result.get('error')}")`}
              />
            </TabsContent>

            <TabsContent value="typescript" className="mt-4">
              <CodeBlock
                id="process-typescript"
                language="typescript"
                code={`const processDocument = async (file: File, template: string) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('template', template);
  formData.append('context', JSON.stringify({
    company_name: 'Acme Corp',
    currency: 'USD'
  }));

  const response = await fetch('${baseUrl}/api/v1/extract/process', {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${YOUR_API_KEY}\`
    },
    body: formData
  });

  const result = await response.json();

  if (result.success) {
    console.log('Job ID:', result.job_id);
    console.log('Extracted data:', result.data);
  }

  return result;
}`}
              />
            </TabsContent>
          </Tabs>

          <div className="mt-4 p-3 bg-muted rounded-lg text-sm">
            <strong>Parameters:</strong>
            <ul className="list-disc list-inside mt-2 space-y-1 text-muted-foreground">
              <li><code>file</code> (required) - Document file (PDF, JPG, PNG, max 10MB)</li>
              <li><code>template</code> (optional) - Template ID (default: detailed_invoice)</li>
              <li><code>context</code> (optional) - JSON string with template variables</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
