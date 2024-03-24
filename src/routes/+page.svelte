<script lang="ts">
	import { writable } from 'svelte/store';
	//   import { goto } from '@sveltejs/kit';
	import { goto } from '$app/navigation';
	import { uploadedImgBase64, uploadedImgFileName } from '../stores/imgStore';
	import { mainWorker } from '../stores/workerStore';
	import { AppBar, AppShell, FileDropzone, LightSwitch } from '@skeletonlabs/skeleton';
	import { FileUp } from 'lucide-svelte';
	import { onMount } from 'svelte';
	import {base} from '$app/paths';

	interface cardProps {
		title: string;
		description: string;
		image: string;
	}
	let w: Worker;
	let cards: cardProps[] = [
		{
			title: 'Step 1: Upload Your Image',
			description: 'Upload or drag and drop an image into the “Upload Image” frame.',
			image: `${base}/img/process_part1.png`
		},
		{
			title: 'Step 2: Select area you want to remove using smart selector tool or brush',
			description:
				'Specify one or more points of the object you want to select using smart selector tool. Alternatively, use a brush tool to select the whole area, or to refine area selected by smart selector.',
			image: `${base}/img/process_part2.png`
		},
		{
			title: 'Step 3: Remove the area',
			description: 'Click the “Remove” button and wait for the result.',
			image: `${base}/img/process_part3.png`
		},
		{
			title: 'Step 4: Download the result image',
			description: 'Compare the result with the original image and download the result image.',
			image: `${base}/img/process_part4.png`
		}
	];

	const handleImageUpload = async (event: Event) => {
		const files = (event.target as HTMLInputElement).files;
		let uploadedImageFile = files?.[0];
		if (!uploadedImageFile) return;
		uploadedImgFileName.set(
			uploadedImageFile.name.substring(0, uploadedImageFile.name.lastIndexOf('.'))
		);

		let reader = new FileReader();
		reader.onload = async (e) => {
			// Store the uploaded image data in the store
			uploadedImgBase64.set(e.target?.result as string);

			// Redirect to the editor page
			goto(`${base}/editor`);
		};

		reader.readAsDataURL(uploadedImageFile);
	};

	onMount(() => {
		if(!$mainWorker){
			w = new Worker(new URL('./../workers/mainworker.worker.js', import.meta.url) , { type: "module" });
			mainWorker.set(w);
		}
	});
</script>

<AppShell>
	<svelte:fragment slot="header">
		<AppBar>
			<svelte:fragment slot="lead">
				<a href={base == '' ? '/' : base} class="font-bold">Smart Object remover</a>
			</svelte:fragment>
			<svelte:fragment slot="trail">
				<a href={base == '' ? '/' : base} class="font-semibold">Home</a>
				<a href="{base}/about" class="font-semibold">About</a>
				<LightSwitch />
			</svelte:fragment>
		</AppBar>
	</svelte:fragment>

	<div>
		<div class="py-4 2xl:px-32 px-8">
			<h1 class="h1 pt-8 font-bold">Remove objects from images with powerful AI tools</h1>
			<h3 class="h3 pt-4">
				Easily remove any unwanted objects, people, defects or text from images with help of
				AI-powered tools
			</h3>
			<div class="md:pt-16 pt-4 flex md:flex-row flex-col-reverse gap-4">
				<div class="flex-1">
					<img src="{base}/img/before_after_example.png" alt="before_after_exampler" />
				</div>
				<FileDropzone
					class="flex-1"
					name="files"
					accept="image/*"
					on:change={(e) => handleImageUpload(e)}
				>
					<svelte:fragment slot="lead">
						<div class="flex justify-center">
							<FileUp size={64} />
						</div>
					</svelte:fragment>
					<svelte:fragment slot="message"
						><span class="font-semibold">Upload an image or drag and drop</span></svelte:fragment
					>
				</FileDropzone>
			</div>

			<h2 class="h2 md:pt-16 md:pb-16 pt-8 pb-8 text-center font-semibold">How does it work?</h2>
			<div class="">
				<div class="w-fit text-token grid grid-cols-1 xl:grid-cols-4 md:grid-cols-2 gap-4">
					{#each cards as card}
						<div class="card overflow-hidden">
							<header>
								<img
									src={card.image}
									class="bg-black/50 w-full h-64 object-cover"
									alt="Object removal process guide"
								/>
							</header>
							<div class="p-8 space-y-4">
								<h3 class="h3 font-semibold">{card.title}</h3>
								<article>
									<p>{card.description}</p>
								</article>
							</div>
						</div>
					{/each}
				</div>
			</div>
		</div>
	</div>
</AppShell>
