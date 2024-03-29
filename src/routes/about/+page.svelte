<script>
	import Navbar from '../../components/navbar.svelte';
	import { base } from '$app/paths';
</script>

<Navbar
	navTitle={{ name: 'Smart Object Remover', href: base == '' ? '/' : base }}
	links={[
		{ name: 'Home', href: base == '' ? '/' : base },
		{ name: 'About', href: `${base}/about` }
	]}
/>
<h1 class="text-3xl font-bold">About</h1>
