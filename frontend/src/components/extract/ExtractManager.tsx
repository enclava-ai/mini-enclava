"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DocumentProcessor } from "./DocumentProcessor"
import { TemplateManager } from "./TemplateManager"

export function ExtractManager() {
  const [activeTab, setActiveTab] = useState("process")

  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
      <TabsList className="grid w-full grid-cols-2 max-w-md">
        <TabsTrigger value="process">Process Documents</TabsTrigger>
        <TabsTrigger value="templates">Manage Templates</TabsTrigger>
      </TabsList>

      <TabsContent value="process" className="mt-6">
        <DocumentProcessor />
      </TabsContent>

      <TabsContent value="templates" className="mt-6">
        <TemplateManager />
      </TabsContent>
    </Tabs>
  )
}
