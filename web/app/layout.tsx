import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GEX Web",
  description: "Hosted GEX service foundation"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth bg-[#09111f]">
      <body className="antialiased min-h-screen text-slate-100">{children}</body>
    </html>
  );
}
