// YouTube site adapter
// Loaded before content.js via manifest content_scripts js array
var youtubeConfig = {
    commentSelector: "#content-text",
    commentContainer: "ytd-comment-thread-renderer",
    delayMs: 4000  // YouTube loads comments async — need a delay
};
