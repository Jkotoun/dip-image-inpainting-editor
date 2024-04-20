<script lang="ts">
	import { goto } from '$app/navigation';
	import { uploadedImgBase64, uploadedImgFileName, uploadedImgTargetRes } from '../stores/imgStore';
	import { mainWorker } from '../stores/workerStore';
	import { browser } from '$app/environment';

	import {
		AppShell,
		FileDropzone,
		getModalStore,
		type ModalSettings
	} from '@skeletonlabs/skeleton';
	import { FileUp, ArrowLeft, ArrowRight } from 'lucide-svelte';
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { MESSAGE_TYPES } from '../workers/messageTypes';
	import Navbar from './../components/Navbar.svelte';
	import EditStepCard from '../components/EditStepCard.svelte';
	let gExamplesContainerElem: HTMLDivElement;

	function multiColumnLeft(): void {
		let x = gExamplesContainerElem.scrollWidth;
		if (gExamplesContainerElem.scrollLeft !== 0)
			x = gExamplesContainerElem.scrollLeft - gExamplesContainerElem.clientWidth;
		gExamplesContainerElem.scroll(x, 0);
	}

	function multiColumnRight(): void {
		let x = 0;
		// -1 is used because different browsers use different methods to round scrollWidth pixels.
		if (
			gExamplesContainerElem.scrollLeft <
			gExamplesContainerElem.scrollWidth - gExamplesContainerElem.clientWidth - 1
		)
			x = gExamplesContainerElem.scrollLeft + gExamplesContainerElem.clientWidth;
		gExamplesContainerElem.scroll(x, 0);
	}
	//high res resize modal
	const modalStore = getModalStore();
	const longsideResResizeThreshold = 2000;

	interface cardProps {
		title: string;
		description: string;
		image: string;
	}
	let w: Worker;
	const cards: cardProps[] = [
		{
			title: 'Step 1: Upload Your Image',
			description: 'Upload or drag and drop an image into the “Upload Image” frame.',
			image: `${base}/img/process_part1.png`
		},
		{
			title: 'Step 2: Select area you want to remove using smart selector tool or brush',
			description:
				'Specify one or more points of the object you want to select using smart selector tool. \
				Alternatively, use a brush tool to select the whole area, or to refine area selected by smart selector. \
				For best results, selected area should contain all edges and shadows of the object',
			image: `${base}/img/process_part2.png`
		},
		{
			title: 'Step 3: Remove the area',
			description: 'Click the “Remove” button and wait for the result.',
			image: `${base}/img/process_part3.png`
		},
		{
			title: 'Step 4: Download the result image',
			description:
				'Compare the result with the original image and repeat the process on other objects or download the result image.',
			image: `${base}/img/process_part4.png`
		}
	];

	const exampleImages = [
		`${base}/example_photos/example_sea_day.webp`,
		`${base}/example_photos/trutnov_trails.jpg`,
		`${base}/example_photos/example_grid.jpg`,
		`${base}/example_photos/example_dog.jpg`,
		`${base}/example_photos/overcast.webp`,
		`${base}/example_photos/example_people_sunset.jpg`,
		`${base}/example_photos/lukas_snih.jpg`,
		`${base}/example_photos/20230808_181343.jpg`,
		`${base}/example_photos/example_sunset2.jpg`,
	];
	// `${base}/example_photos/free-photo-of-ryma-studeny-snih-drevo.jpeg`,
	// `${base}/example_photos/mountain-5649827_1280.jpg`,
	// `${base}/example_photos/1920resized_street.jpg`
	
	async function getImageResolution(url: string): Promise<{ width: number; height: number }> {
		const img = document.createElement('img');
		const promise: Promise<{ width: number; height: number }> = new Promise((resolve, reject) => {
			img.onload = () => {
				const width = img.naturalWidth;
				const height = img.naturalHeight;
				resolve({ width, height });
			};

			img.onerror = reject;
		});

		img.src = url;
		return promise;
	}
	//https://stackoverflow.com/questions/22172604/convert-image-from-url-to-base64
	async function getImgDataFromUrl(
		url: string
	): Promise<{
		base64data: string;
		imageName: string;
		resolution: { width: number; height: number };
	}> {
		const urlParts = url.split('/');
		const imageName: string = urlParts[urlParts.length - 1];

		const data = await fetch(url);
		const blob = await data.blob();

		const img = new Image();
		img.src = URL.createObjectURL(blob);
		const promise: any = new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.readAsDataURL(blob);
			reader.onloadend = () => {
				const base64data: string = reader.result as string;
				img.onload = () => {
					const resolution: { width: number; height: number } = {
						width: img.width,
						height: img.height
					};
					resolve({ base64data: base64data, imageName: imageName, resolution });
				};

				// If there's an error loading the image, reject the promise
				img.onerror = () => {
					reject('Error loading image');
				};
			};
			reader.onerror = reject;
		});

		return promise;
	}

	//on user image upload, redirect to editor page and store img data to store
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

			const { width, height } = await getImageResolution(e.target?.result as string);
			uploadedImgTargetRes.set({ width, height });
			if (width > longsideResResizeThreshold || height > longsideResResizeThreshold) {
				let aspectRatio = width / height;
				let newWidth =
					aspectRatio > 1
						? longsideResResizeThreshold
						: Math.round(longsideResResizeThreshold * aspectRatio);
				let newHeight =
					aspectRatio > 1
						? Math.round(longsideResResizeThreshold / aspectRatio)
						: longsideResResizeThreshold;
				const highResResizeModal: ModalSettings = {
					type: 'confirm',
					title: 'High resolution image resize',
					body: `The image "${$uploadedImgFileName}" has very high resolution of ${width}x${height}, which may cause poor performance of editor. \
					Do you want to resize the image to ${newWidth}x${newHeight} for better performance or keep the best image \
					quality despite possible performance issues?`,
					buttonTextCancel: `Keep original resolution`,
					buttonTextConfirm: `Resize to ${newWidth}x${newHeight}`,
					response: (resize) => {
						if (resize) {
							uploadedImgTargetRes.set({ width: newWidth, height: newHeight });
						}
						goto(`${base}/editor`);
					}
				};
				modalStore.trigger(highResResizeModal);
			} else {
				goto(`${base}/editor`);
			}
		};
		reader.readAsDataURL(uploadedImageFile);
	};

	//init worker
	onMount(() => {
		if (!$mainWorker) {
			w = new Worker(new URL('./../workers/mainworker.worker.js', import.meta.url), {
				type: 'module'
			});
			w.postMessage({
				type: MESSAGE_TYPES.INIT,
				data: {
					env: process.env.NODE_ENV,
					appBasePath: base
				}
			});
			mainWorker.set(w);
		}
		uploadedImgBase64.set(null);
		uploadedImgFileName.set('');
		uploadedImgTargetRes.set(null);
	});
</script>

<AppShell>
	<svelte:fragment slot="header">
		<Navbar
			navTitle={{ name: 'Smart Object Remover', href: base == '' ? '/' : base }}
			links={[{ name: 'Home', href: base == '' ? '/' : base }]}
		/>
	</svelte:fragment>

	<div>
		<!-- <button on:click={() => }>asda</button> -->

		<div class="py-4 2xl:px-24 px-8">
			<h1 class="h1 pt-8 font-bold">Remove objects from images with powerful AI tools</h1>
			<h3 class="h3 pt-4">
				Easily remove any unwanted objects, people, defects or text from images with help of
				AI-powered tools
			</h3>
			<div class="md:pt-16 pt-4 flex md:flex-row flex-col-reverse gap-4">
				<div class="flex-1">
					<img src="{base}/img/before_after_example.png" alt="before_after_exampler" />
				</div>
				<div class="flex-1  ">
					
						<FileDropzone name="files" padding="2xl:py-16 xl:py-12 lg:py-8 md:py-2 py-8 " accept="image/*" on:change={(e) => handleImageUpload(e)}>
							<svelte:fragment slot="lead">
								<div class="flex justify-center">
									<FileUp size={64} />
								</div>
							</svelte:fragment>
							<svelte:fragment slot="message"
								><span class="font-semibold">Upload an image or drag and drop</span
								></svelte:fragment
							>
						</FileDropzone>
					
					<h4 class="h4 font-semibold pt-4 pb-2 text-center">Or try with an example</h4>
					<div class="grid grid-cols-[auto_1fr_auto] gap-4 items-center  pt-4">
						<!-- Button: Left -->
						<button type="button" class="btn-icon btn-icon-sm variant-filled" on:click={multiColumnLeft}>
							<ArrowLeft size={20}/>
						</button>
						<!-- Carousel -->
						<div
							bind:this={gExamplesContainerElem}
							class="snap-x snap-mandatory  max-h-[125px]   scroll-smooth flex gap-5 pb-2 overflow-hidden "
						>
							{#each exampleImages as imgpath}
								<button
									type="button"
									class="shrink-0 w-[28%] snap-start"
									on:click={async (e) => {
										const { base64data, imageName, resolution } = await getImgDataFromUrl(imgpath);
										uploadedImgBase64.set(base64data);
										uploadedImgFileName.set(imageName);
										uploadedImgTargetRes.set(resolution);
										goto(`${base}/editor`);
									}}
								>
									<img class="cursor-pointer  object-cover object-center" src={imgpath} alt="example_image" />
								</button>
							{/each}
						</div>
						<!-- Button-Right -->
						<button type="button" class="btn-icon btn-icon-sm variant-filled" on:click={multiColumnRight}>
							<ArrowRight size={20} />
						</button>
					</div>
				</div>
			</div>

			<h2 class="h2 md:pt-16 md:pb-16 pt-8 pb-8 text-center font-semibold">How does it work?</h2>
			<div class="w-fit text-token grid grid-cols-1 xl:grid-cols-4 md:grid-cols-2 gap-4">
				{#each cards as card}
					<EditStepCard title={card.title} description={card.description} imgSrc={card.image} />
				{/each}
			</div>
		</div>
	</div>
</AppShell>


<style>
	.dropzone{
		height: 100% !important
	} 
</style>