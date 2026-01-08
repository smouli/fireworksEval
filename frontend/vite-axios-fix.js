// Vite plugin to fix axios internal module resolution
export default function axiosFix() {
  return {
    name: 'axios-fix',
    async resolveId(source, importer) {
      // Fix axios internal relative imports
      if (source === './env/data.js' && importer?.includes('axios/lib/axios.js')) {
        // Resolve using Vite's resolver
        const resolved = await this.resolve('axios/lib/env/data.js', importer, { skipSelf: true })
        return resolved || null
      }
      return null
    },
  }
}

