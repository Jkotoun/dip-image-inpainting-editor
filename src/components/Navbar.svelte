<script lang="ts">
	import { AppBar, LightSwitch } from '@skeletonlabs/skeleton';
	import { createEventDispatcher } from 'svelte';
	const dispatch = createEventDispatcher();
	export let basepath: string;
	export let drawerMenu: boolean = false;
	export let navTitle: { name: string; href?: string };
	export let links: { name: string; href: string }[] = [];
	//
</script>

<AppBar>
	<svelte:fragment slot="lead">
		{#if drawerMenu}
			<button class="lg:hidden btn btn-sm mr-4" on:click={() => dispatch('openDrawerMenu')}>
				<span>
					<svg viewBox="0 0 100 80" class="fill-token w-4 h-4">
						<rect width="100" height="20" />
						<rect y="30" width="100" height="20" />
						<rect y="60" width="100" height="20" />
					</svg>
				</span>
			</button>
		{/if}
		{#if navTitle.href}
			<a href={navTitle.href} class="font-bold lg:inline hidden">{navTitle.name}</a>
		{:else}
			<span class="font-bold lg:inline hidden">{navTitle.name}</span>
		{/if}
	</svelte:fragment>
	<svelte:fragment slot="trail">
		{#each links as link}
			<a href={link.href} class="font-semibold">{link.name}</a>
		{/each}
		<LightSwitch />
	</svelte:fragment>
</AppBar>

<style>
</style>
