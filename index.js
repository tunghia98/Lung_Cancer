// Function to diagnose uploaded image
function diagnoseImage() {
    var formData = new FormData(document.getElementById('uploadForm'));
    var fileInput = document.getElementById('imageFile');
    fetch('http://localhost:8000/diagnose/', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (fileInput.files && fileInput.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                uploadedImage.src = e.target.result;
                Swal.fire({
                    imageUrl: e.target.result,
                    title: data.prediction
                  });
            };
            reader.readAsDataURL(fileInput.files[0]);
        }
    })
    .catch(error => {
        Swal.fire({
            title: "<strong class='error'>Error:</strong> " + error
          });
        console.error('Error:', error);
    });
}

// Function to fetch image directories
function getImageDirectories(split) {
    fetch(`http://localhost:8000/image-directories/${split}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById('imageDirectories').innerHTML = `<strong>${split} Directories:</strong> ${data}`;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('imageDirectories').innerHTML = "<strong class='error'>Error:</strong> " + error;
    });
}

// Function to fetch metrics
function getMetrics() {
    fetch('http://localhost:8000/metrics/')
    .then(response => response.json())
    .then(data => {
        document.getElementById('metrics').innerHTML = "<strong>Metrics:</strong><br>" + JSON.stringify(data, null, 2);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('metrics').innerHTML = "<strong class='error'>Error:</strong> " + error;
    });
}