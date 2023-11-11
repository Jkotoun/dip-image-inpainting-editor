<script>
    let a = 10;
    $: b = 2 * a
</script>
<h1>Welcome to SvelteKit</h1>
<p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>
<p>{a}</p>
<p>{b}</p>

<button on:click={() => a++}>Increment</button>


<a href="/model_demo">model demo</a>