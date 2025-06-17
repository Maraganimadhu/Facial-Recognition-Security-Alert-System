async function captureImage() {
    const video = document.getElementById('videoFeed');
    if (!video || !video.src) {
        console.error("Video feed not available");
        return null;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = video.width;
    canvas.height = video.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

document.getElementById('registerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = e.target.querySelector('input[name="name"]').value;
    const image = await captureImage();
    
    if (!image) {
        document.getElementById('registerResult').innerHTML = "Error: Camera not accessible";
        return;
    }
    
    const formData = new FormData();
    formData.append('name', name);
    formData.append('image', image);
    
    const response = await fetch('/register', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('registerResult').innerHTML = 
        response.ok ? result.message : result.error;
});

document.getElementById('attendanceForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const image = await captureImage();
    
    if (!image) {
        document.getElementById('attendanceResult').innerHTML = "Error: Camera not accessible";
        return;
    }
    
    const formData = new FormData();
    formData.append('image', image);
    
    const response = await fetch('/mark_attendance', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    document.getElementById('attendanceResult').innerHTML = 
        response.ok ? `${result.message} at ${result.timestamp}` : result.error;
});