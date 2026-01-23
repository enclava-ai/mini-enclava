"use client"

import { useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { DocumentProcessor } from "./DocumentProcessor"
import { TemplateManager } from "./TemplateManager"
import { IntegrationGuide } from "./IntegrationGuide"
import { ExtractSettings } from "./ExtractSettings"

export function ExtractManager() {
  const [activeTab, setActiveTab] = useState("process")

  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
      <TabsList className="grid w-full grid-cols-4 max-w-3xl">
        <TabsTrigger value="process">Process Documents</TabsTrigger>
        <TabsTrigger value="templates">Manage Templates</TabsTrigger>
        <TabsTrigger value="settings">Settings</TabsTrigger>
        <TabsTrigger value="integration">API Integration</TabsTrigger>
      </TabsList>

      <TabsContent value="process" className="mt-6">
        <DocumentProcessor />
      </TabsContent>

      <TabsContent value="templates" className="mt-6">
        <TemplateManager />
      </TabsContent>

      <TabsContent value="settings" className="mt-6">
        <ExtractSettings />
      </TabsContent>

      <TabsContent value="integration" className="mt-6">
        <IntegrationGuide />
      </TabsContent>
    </Tabs>
  )
}
