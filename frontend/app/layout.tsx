import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Enterprise AI — Intelligence Platform",
  description: "AI-powered enterprise knowledge base with RAG, agents, and evaluation",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#0f1117] min-h-screen">
        <AuthProvider>
          <Sidebar />
          <div id="main-content">
            {children}
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}
