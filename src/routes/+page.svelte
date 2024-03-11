<script lang="ts">
	import { writable } from 'svelte/store';
	//   import { goto } from '@sveltejs/kit';
	import { goto } from '$app/navigation';
    import { uploadedImgBase64, uploadedImgFileName } from '../stores';
	// let uploadedImageData = writable("");

	const handleImageUpload = async (event: Event) => {
        const files = (event.target as HTMLInputElement).files;
		let uploadedImageFile = files?.[0];
		if (!uploadedImageFile) return;
		uploadedImgFileName.set(uploadedImageFile.name.substring(0, uploadedImageFile.name.lastIndexOf('.')))

		let reader = new FileReader();
		reader.onload = async (e) => {
			// Store the uploaded image data in the store
			uploadedImgBase64.set(e.target?.result as string);

			// Redirect to the editor page
			goto('/editor');
		};

		reader.readAsDataURL(uploadedImageFile);
	};
</script>

<h1>AI object remover</h1>
<div>
	<input type="file" accept="image/*" on:change={(e) => handleImageUpload(e)} />
</div>
<!-- <a href="/editor">Editor</a> -->
<br />
