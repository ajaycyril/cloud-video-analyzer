import { ClientApp } from "@/components/ClientApp";

export const dynamic = "force-dynamic";

export default function Home() {
  return (
    <ClientApp
      providerStatus={{
        gemini: Boolean(process.env.GEMINI_API_KEY),
        openai: Boolean(process.env.OPENAI_API_KEY),
        nvidia: Boolean(process.env.NVIDIA_API_KEY),
      }}
    />
  );
}
