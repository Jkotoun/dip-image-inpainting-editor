<script>
	import Navbar from '../components/Navbar.svelte';
    import './../app.css';
	import { AppShell, AppBar } from '@skeletonlabs/skeleton';

</script>

<Navbar title="AI Object Remover" items={
	[
		{title: "Home", link: "/"},
		// {title: "About", link: "/about", selected:false},
	]
} />
<slot/>
