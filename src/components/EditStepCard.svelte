<script lang="ts">
    export let imgSrc: string;
    export let title: string;
    export let description: string;
</script>
<div class="card overflow-hidden drop-shadow-sm border">
    <header>
        <img
            src={imgSrc}
            class="bg-black/50 w-full h-64 object-cover"
            alt="Object removal process guide"
        />
    </header>
    <div class="p-8 space-y-4">
        <h3 class="h3 font-semibold">{title}</h3>
        <article>
            <p>{description}</p>
        </article>
    </div>
</div>