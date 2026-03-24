// Instagram site adapter
// Loaded before content.js via manifest content_scripts js array
var instagramConfig = {
    // Instagram renders comments inside nested <ul> > <li> > <span> elements
    commentSelector: "ul ul li span span",
    commentContainer: "ul ul li",
    delayMs: 4000
};
