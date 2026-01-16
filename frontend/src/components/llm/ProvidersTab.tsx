'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/lib/api-client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, XCircle, RefreshCw } from 'lucide-react';

interface ProviderHealth {
  provider_id: string;
  display_name: string;
  healthy: boolean;
  last_check_at: string | null;
  last_healthy_at: string | null;
  error: string | null;
  attestation_details: {
    intel_tdx_verified: boolean;
    gpu_attestation_verified: boolean;
    nonce_binding_verified: boolean;
    signing_address: string | null;
  } | null;
  // Pricing from TOKEN_STATS_PLAN.md
  pricing: {
    source: string;
    last_sync_at: string | null;
    model_count: number;
  } | null;
}

export default function ProvidersTab() {
  const [providers, setProviders] = useState<ProviderHealth[]>([]);
  const [loading, setLoading] = useState(true);
  const [verifying, setVerifying] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchProviders();
    const interval = setInterval(fetchProviders, 30000);
    return () => clearInterval(interval);
  }, []);

  async function fetchProviders() {
    try {
      setError(null);
      const data = await apiClient.get('/api-internal/v1/providers/health');
      setProviders(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch providers');
    } finally {
      setLoading(false);
    }
  }

  async function verifyProvider(providerId: string) {
    setVerifying(providerId);
    try {
      await apiClient.post(`/api-internal/v1/providers/${providerId}/verify`);
      await fetchProviders();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed');
    } finally {
      setVerifying(null);
    }
  }

  function formatTimeAgo(dateStr: string | null): string {
    if (!dateStr) return 'Never';
    const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-empire-gold"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Inference Providers</h2>
          <p className="text-muted-foreground text-sm">
            Confidential AI providers with TEE attestation.
          </p>
        </div>
        <Button
          variant="outline"
          onClick={fetchProviders}
          disabled={loading}
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {error && (
        <div className="p-3 bg-destructive/10 rounded-md text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="space-y-4">
        {providers.map((provider) => (
          <Card key={provider.provider_id}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-lg">{provider.display_name}</CardTitle>
              {provider.healthy ? (
                <Badge className="bg-green-500 hover:bg-green-600 text-white border-transparent">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Healthy
                </Badge>
              ) : (
                <Badge variant="destructive">
                  <XCircle className="w-3 h-3 mr-1" />
                  Unhealthy
                </Badge>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-sm text-muted-foreground">
                Last checked: {formatTimeAgo(provider.last_check_at)}
              </div>

              {/* Pricing info (from TOKEN_STATS_PLAN.md) */}
              {provider.pricing && (
                <div className="text-sm text-muted-foreground">
                  Pricing: {provider.pricing.source === 'api_sync' ? (
                    <>Auto-sync (last synced {formatTimeAgo(provider.pricing.last_sync_at)}, {provider.pricing.model_count} models)</>
                  ) : (
                    <>Manual ({provider.pricing.model_count} models)</>
                  )}
                </div>
              )}

              {/* Error message */}
              {!provider.healthy && provider.error && (
                <div className="p-3 bg-destructive/10 rounded-md text-sm text-destructive">
                  {provider.error}
                </div>
              )}

              {/* Attestation details (for RedPill) */}
              {provider.healthy && provider.attestation_details && (
                <div className="text-sm space-y-1">
                  <div className="flex gap-4">
                    <span>{provider.attestation_details.intel_tdx_verified ? '✓' : '✗'} Intel TDX</span>
                    <span>{provider.attestation_details.gpu_attestation_verified ? '✓' : '✗'} NVIDIA GPU</span>
                    <span>{provider.attestation_details.nonce_binding_verified ? '✓' : '✗'} Nonce</span>
                  </div>
                  {provider.attestation_details.signing_address && (
                    <div className="text-muted-foreground">
                      Signer: {provider.attestation_details.signing_address.slice(0, 10)}...
                    </div>
                  )}
                </div>
              )}

              <Button
                variant="outline"
                size="sm"
                onClick={() => verifyProvider(provider.provider_id)}
                disabled={verifying === provider.provider_id}
              >
                {verifying === provider.provider_id ? 'Verifying...' : 'Verify Now'}
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
