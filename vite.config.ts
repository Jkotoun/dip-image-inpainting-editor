import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
//@ts-ignore
import crossOriginIsolation from 'vite-plugin-cross-origin-isolation'

export default defineConfig({
	plugins: [
		crossOriginIsolation(),
		sveltekit(),

	],
});
