<script lang="ts">
    interface Item {
        title: string;
        link: string;
        selected: boolean;
    }
    export let title: string;
    export let items: Item[];
</script>


<navbar>

    <div class="navbar">
        <div class="navbar-title">
            <h1>{title}</h1>
        </div>
        <div class="navbar-items">
            {#each items as item}
                <a href={item.link} class:active={item.selected}>{item.title}</a>
            {/each}
        </div>
    </div>

</navbar>


<style>
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #f1f1f1;
    }
    .navbar-title h1 {
        margin: 0;
    }
    .navbar-items a {
        margin: 0 1rem;
        text-decoration: none;
        color: #000;
    }
    .navbar-items a.active {
        color: #f00;
    }
</style> 