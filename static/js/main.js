// Main JavaScript for Leprosy Detection Application

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Initialize all popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
    
    // Custom file input for upload form
    const fileInput = document.querySelector('.custom-file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0].name;
            const nextSibling = this.nextElementSibling;
            nextSibling.innerText = fileName;
            
            // Preview image if it's an image upload
            const previewContainer = document.getElementById('image-preview-container');
            if (previewContainer && this.files && this.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewContainer.innerHTML = `<img src="${e.target.result}" class="img-fluid rounded preview-image" alt="Image Preview">`;
                    previewContainer.classList.remove('d-none');
                }
                
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
    
    // Fade out flash messages after 5 seconds
    const flashMessages = document.querySelectorAll('.alert:not(.alert-permanent)');
    if (flashMessages.length > 0) {
        setTimeout(function() {
            flashMessages.forEach(function(message) {
                const alert = new bootstrap.Alert(message);
                alert.close();
            });
        }, 5000);
    }
});