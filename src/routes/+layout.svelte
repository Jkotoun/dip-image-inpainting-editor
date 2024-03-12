<script>
	import Navbar from '../components/Navbar.svelte';
	import './../app.css';
	import { AppShell, AppBar } from '@skeletonlabs/skeleton';
</script>

<!-- <Navbar title="AI Object Remover" items={
	[
		{title: "Home", link: "/"},
		// {title: "About", link: "/about", selected:false},
	]
} /> -->
<AppShell>
	<svelte:fragment slot="header">
		<AppBar>
			<svelte:fragment slot="lead">
				<a href="/" class="font-bold">Smart Object remover</a>
			</svelte:fragment>
			<svelte:fragment slot="trail">
				<a href="/">Home</a>
				<a href="/about">About</a>
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>
</AppShell>
<slot />
