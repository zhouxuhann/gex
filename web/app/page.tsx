import Link from 'next/link';

export default function LandingPage() {
  return (
    <main className="relative flex flex-col min-h-screen overflow-hidden">
      {/* Background Effects */}
      <div className="gex-shell fixed inset-0 pointer-events-none z-0" />
      <div className="gex-grid-fade fixed inset-0 pointer-events-none z-0" />
      
      {/* Hero Section */}
      <section className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 text-center">
        <div className="max-w-4xl space-y-8 animate-fade-in-up">
          <div className="inline-flex items-center rounded-full border border-white/10 bg-white/5 py-1.5 pl-3 pr-4 text-sm font-medium text-[#6ee7ff] backdrop-blur-md">
            <span className="relative flex h-2.5 w-2.5 mr-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#6ee7ff] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#6ee7ff]"></span>
            </span>
            Real-time Institutional Data
          </div>
          
          <h1 className="text-6xl font-extrabold tracking-tight sm:text-7xl lg:text-8xl bg-clip-text text-transparent bg-gradient-to-br from-white via-slate-100 to-slate-500 pb-2">
            Decode The <br/> Market Engine
          </h1>
          
          <p className="mx-auto max-w-2xl text-lg text-slate-300 sm:text-xl leading-relaxed animate-fade-in-up delay-150">
            Professional Gamma Exposure Analytics. We translate complex dealer positioning into actionable visual intelligence for the modern trader.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-5 pt-6 animate-fade-in-up delay-300">
            <Link 
              href="/signup" 
              className="w-full sm:w-auto relative group overflow-hidden rounded-full p-[1px] transition-transform hover:scale-105 active:scale-95"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-cyan-400 to-[#6ee7ff] rounded-full" />
              <div className="relative flex items-center justify-center rounded-full bg-[#09111f] px-8 py-3.5 font-semibold text-white transition-all group-hover:bg-transparent shadow-[0_0_20px_rgba(110,231,255,0.2)] group-hover:shadow-[0_0_30px_rgba(110,231,255,0.4)]">
                Become a Member
              </div>
            </Link>
            
            <Link 
              href="/login" 
              className="w-full sm:w-auto rounded-full border border-white/10 bg-white/5 px-8 py-3.5 font-semibold text-white backdrop-blur-md transition-all hover:bg-white/10 hover:border-white/20 hover:text-[#6ee7ff] active:scale-95 shadow-xl"
            >
              Sign In
            </Link>
          </div>
        </div>

        {/* Scroll Indicator */}
        <div className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 animate-fade-in-up delay-500">
          <Link href="#intro" className="flex flex-col items-center text-slate-400 hover:text-white transition-colors group text-sm font-medium tracking-wide">
            Learn what you can do with our GEX
            <div className="mt-4 rounded-full border border-white/10 bg-white/5 p-3 group-hover:bg-white/10 group-hover:border-white/20 transition-all group-hover:translate-y-1">
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m6 9 6 6 6-6"/>
              </svg>
            </div>
          </Link>
        </div>
      </section>

      {/* Intro Section */}
      <section id="intro" className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-24">
        <div className="max-w-6xl w-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-24 items-center">
            
            {/* Abstract Visual Dashboard */}
            <div className="relative order-2 lg:order-1 w-full aspect-square md:aspect-[4/3] lg:aspect-square rounded-[2rem] border border-white/10 bg-gradient-to-br from-white/5 to-transparent p-1 shadow-2xl backdrop-blur-xl overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-tr from-[#6ee7ff]/10 via-transparent to-blue-500/10 opacity-50 group-hover:opacity-100 transition-opacity duration-700" />
              
              <div className="relative w-full h-full bg-[#09111f]/80 rounded-[2rem] border border-white/5 overflow-hidden flex flex-col p-6 shadow-inner">
                {/* Mock Header */}
                <div className="flex items-center justify-between border-b border-white/10 pb-4 mb-6">
                  <div className="flex items-center gap-3">
                    <div className="h-3 w-3 rounded-full bg-red-500/80" />
                    <div className="h-3 w-3 rounded-full bg-yellow-500/80" />
                    <div className="h-3 w-3 rounded-full bg-green-500/80" />
                  </div>
                  <div className="h-4 w-24 rounded bg-white/10" />
                </div>
                
                {/* Mock Chart Area */}
                <div className="flex-1 relative flex items-end justify-between gap-2 px-2 pb-8">
                  {/* Grid Lines */}
                  <div className="absolute inset-0 flex flex-col justify-between pt-2 pb-8 pointer-events-none">
                    {[...Array(5)].map((_, i) => (
                      <div key={i} className="w-full h-px bg-white/5" />
                    ))}
                  </div>
                  
                  {/* Bars */}
                  {[
                    { h: '30%', c: 'from-red-500/80 to-red-500/20' },
                    { h: '15%', c: 'from-red-500/80 to-red-500/20' },
                    { h: '8%',  c: 'from-slate-500/40 to-slate-500/10' },
                    { h: '25%', c: 'from-green-500/80 to-[#6ee7ff]/20' },
                    { h: '75%', c: 'from-green-400 to-[#6ee7ff]/40 shadow-[0_0_20px_rgba(110,231,255,0.4)]' }, // Peak Call Wall
                    { h: '45%', c: 'from-green-500/80 to-[#6ee7ff]/20' },
                    { h: '20%', c: 'from-green-500/80 to-[#6ee7ff]/20' },
                  ].map((bar, i) => (
                    <div key={i} className={`relative w-full rounded-t flex-1 bg-gradient-to-t ${bar.c} transition-all duration-1000 transform origin-bottom hover:scale-110`} style={{ height: bar.h }}>
                      {i === 4 && (
                        <div className="absolute -top-8 left-1/2 -translate-x-1/2 rounded bg-white/10 px-2 py-1 text-[10px] font-medium text-white whitespace-nowrap backdrop-blur-md border border-white/20">
                          Call Wall
                        </div>
                      )}
                    </div>
                  ))}
                  
                  {/* Horizon Line / Zero Gamma */}
                  <div className="absolute bottom-8 left-0 right-0 h-px bg-yellow-400/50 flex flex-col items-start justify-center group-hover:bg-yellow-400 transition-colors">
                    <div className="flex items-center gap-2 translate-y-3 px-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-yellow-400 animate-pulse" />
                      <span className="text-[10px] text-yellow-400/80 font-mono tracking-wider">GAMMA FLIP</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Text Content */}
            <div className="order-1 lg:order-2 space-y-8 animate-fade-in-up">
              <div className="space-y-4">
                <h2 className="text-4xl font-bold tracking-tight sm:text-5xl text-white">
                  Visualize <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-[#6ee7ff]">Gamma Exposure</span> in Real-Time
                </h2>
                <div className="h-1 w-20 bg-gradient-to-r from-blue-500 to-[#6ee7ff] rounded-full" />
              </div>
              
              <div className="space-y-6 text-lg text-slate-400 leading-relaxed max-w-xl">
                <p>
                  Our engine translates complex option chains into intuitive visual metrics. We compute <strong className="text-white">Total GEX, Gamma Flips, and Market Walls</strong> so you don't have to guess where dealer support and resistance lie.
                </p>
                <p>
                  By tracking the hidden forces of market maker hedging, you can anticipate volatility compressions and aggressive expansions long before they appear on traditional price charts.
                </p>
                <ul className="space-y-3 pt-2">
                  {[
                    "Pinpoint exact dealer gamma levels",
                    "Monitor real-time regime shifts",
                    "Identify heavy options interest clusters"
                  ].map((item, i) => (
                    <li key={i} className="flex items-center gap-3">
                      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-green-500/20 text-green-400">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                      </div>
                      <span className="text-slate-200 font-medium">{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="pt-6">
                <Link 
                  href="/intro" 
                  className="inline-flex items-center gap-3 rounded-full bg-white/5 px-8 py-4 font-semibold text-white border border-white/10 transition-all hover:bg-white/10 hover:pr-6 group shadow-lg backdrop-blur-md"
                >
                  Explore Detailed Documentation
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:translate-x-1.5 text-[#6ee7ff]">
                    <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
                  </svg>
                </Link>
              </div>
            </div>
            
          </div>
        </div>
      </section>
    </main>
  );
}
