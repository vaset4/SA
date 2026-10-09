import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(__dirname, '..')

const SOURCE_DIRS = ['app', 'components', 'stores', 'lib']
const DIRECT_FETCH_ALLOWLIST = new Set([
  'lib/api/core.js',
  'lib/auth-client.js',
  'lib/study-assistant-stream.js',
])

function toPosix(relativePath) {
  return relativePath.split(path.sep).join('/')
}

function walkSourceFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  return entries.flatMap((entry) => {
    if (entry.name === 'node_modules' || entry.name === '.next') return []
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) return walkSourceFiles(fullPath)
    return /\.(jsx?|mjs)$/.test(entry.name) ? [fullPath] : []
  })
}

function allSourceFiles() {
  return SOURCE_DIRS
    .map((dir) => path.resolve(root, dir))
    .filter((dir) => fs.existsSync(dir))
    .flatMap((dir) => walkSourceFiles(dir))
}

function relativeSourcePath(filePath) {
  return toPosix(path.relative(root, filePath))
}

function directFetchViolations() {
  return allSourceFiles()
    .map((filePath) => ({
      relativePath: relativeSourcePath(filePath),
      source: fs.readFileSync(filePath, 'utf8'),
    }))
    .filter(({ relativePath, source }) =>
      !DIRECT_FETCH_ALLOWLIST.has(relativePath) && /\bfetch\s*\(/.test(source),
    )
    .map(({ relativePath }) => relativePath)
}

function emptyCatchViolations() {
  return allSourceFiles()
    .map((filePath) => ({
      relativePath: relativeSourcePath(filePath),
      source: fs.readFileSync(filePath, 'utf8'),
    }))
    .filter(({ source }) => /catch\s*(?:\([^)]*\))?\s*\{\s*\}/m.test(source))
    .map(({ relativePath }) => relativePath)
}

test('frontend backend requests are owned by lib/api wrappers or documented transport clients', () => {
  assert.deepEqual(directFetchViolations(), [])
})

test('frontend async failures are not hidden behind empty catch blocks', () => {
  assert.deepEqual(emptyCatchViolations(), [])
})
