// workerStore.js
import { writable } from 'svelte/store';
export const mainWorker = writable<Worker|null>(null);