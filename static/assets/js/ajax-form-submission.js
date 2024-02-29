$(document).ready(function () {
    $('#your-form').submit(function (e) {
        e.preventDefault(); // Prevent the form from submitting traditionally

        // Show loader
        document.getElementById("loader").style.display = "block";
        // Hide the main content
        document.getElementById("main-content").style.display = "none";

        // Perform AJAX submission
        $.ajax({
            type: 'POST',
            url: $('#your-form').attr('action'),
            data: $('#your-form').serialize(),
            success: function (data) {
                // On success, navigate to the next page or perform other actions
                window.location.href = '{% url '' %}';
            },
            error: function (data) {
                // Handle errors if needed
                console.log(data);
            }
        });
    });
});
