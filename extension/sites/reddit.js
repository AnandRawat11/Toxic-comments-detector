// Reddit site adapter
// Loaded before content.js via manifest content_scripts js array
var redditConfig = {
    // New Reddit (2024+): comment body text lives inside a <p> inside the comment element
    commentSelector: "div[data-testid='comment'] p, shreddit-comment p",
    commentContainer: "div[data-testid='comment'], shreddit-comment",
    delayMs: 3000
};
