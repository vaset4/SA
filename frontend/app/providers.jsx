'use client'

import { HeroUIProvider, ToastProvider } from '@heroui/react'
import { ThemeProvider as NextThemesProvider } from 'next-themes'
import { AuthStoreProvider } from '../stores/auth-store'

export function Providers({ children }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="light">
      <HeroUIProvider>
        <AuthStoreProvider>{children}</AuthStoreProvider>
        <ToastProvider placement="bottom-right" />
      </HeroUIProvider>
    </NextThemesProvider>
  )
}
