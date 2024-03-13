<script lang="ts">
	import { Brush, WandSparkles } from 'lucide-svelte';
	import Navbar from '../components/Navbar.svelte';
	import './../app.css';
	import { AppShell, AppBar, TabGroup, Tab } from '@skeletonlabs/skeleton';
	import { LightSwitch } from '@skeletonlabs/skeleton';
</script>


<!-- svelte-ignore missing-declaration -->
<AppShell>
	<svelte:fragment slot="header">
		<AppBar>
			<svelte:fragment slot="lead">
				<a href="/" class="font-bold">Smart Object remover</a>
			</svelte:fragment>
			<svelte:fragment  slot="trail">
				<a href="/" class="font-semibold">Home</a>
				<a href="/about" class="font-semibold">About</a>
				<LightSwitch />
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>
	<!-- <svelte:fragment slot="footer">
		<p class="text-center text-sm font-semibold">
			© 2024 Smart Object Remover
		</p>
	</svelte:fragment> -->
	<slot  />
</AppShell>

