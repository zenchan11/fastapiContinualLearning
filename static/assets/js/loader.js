document.onreadystatechange = function () {
    if (document.readyState === "complete") {
        // Hide loader when the page is fully loaded
        document.getElementById("loader").style.display = "none";
        // Show the main content
        document.getElementById("main-content").style.display = "block";
    } else {
        // Show loader while the page is loading
        document.getElementById("loader").style.display = "block";
        // Hide the main content
        document.getElementById("main-content").style.display = "none";
    }
};
