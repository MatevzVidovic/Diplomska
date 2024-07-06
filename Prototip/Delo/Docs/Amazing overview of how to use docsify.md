
# Amazing simple documentation hosting

## In short
- make a README.md, which will be the homepage at /
- init:
```
npm install -g docsify-cli
docsify init ./docs
```
- Put your .md files into /docs.
- Make _sidebar.md to make the file structure, like so:
```
	- [Home](/)
	- [Introduction](introduction.md)
	- [Installation](installation.md)
	- [Usage](usage.md)
	- [FAQ](faq.md)
	- Guide
	  - [Guide A](guide/guide-a.md)
	  - [Guide B](guide/guide-b.md)
```
- In index.html:
```
<script>
    window.$docsify = {
        ...other stuff
        loadSidebar: true,
        search: 'auto', // enables the search plugin
    }
</script>
<script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
```
- Run it locally and open localhost:3000
```
docsify serve docs
Or if inside /docs in the terminal:
docsify serve .
```


\
\
\
\
\
\
\

## This is all an editing of gpt4 answers on how to set this up both locally and on github pages.


# 0 Awesome to know:
Inside the markdown document you can make some awesome stuff.
If you just move text by a tab or 4 spaces to the right and make the line before it an empty line, it will be a markup field, like so:

    This should be markup.
	This is the same markup, despite having a tab.
	    This one has a tab and 4 spaces and is still the same field.
If you use the same syntax as the _sidebar.md, you will create a text link to another side.
[This](introduction.md) is a link.
To add code, simply use starting and closing three backticks.
```
This is code.
```
This code should even have a copy button if it is hosted on Github Pages.

Reminder that a single backslash means a newline in md.


# 1
Docsify is a tool that dynamically generates a documentation website from your Markdown files. Here’s how to set it up:

Install Docsify: First, install Docsify globally via npm:

	npm install -g docsify-cli

Initialize Docsify in your repository: Navigate to the root of your local repository and run:

	docsify init ./docs

This will create an index.html, a .nojekyll file to prevent GitHub Pages from using Jekyll, and a README.md in the docs folder.

    index.html: The entry point of the site.
    README.md: The main content file, which acts as the homepage.
    .nojekyll: Prevents GitHub Pages from ignoring files that begin with an underscore.




# 2
Write your documentation in separate Markdown files. Organize these files into a directory structure that makes sense for your project, for example:

/docs
  - introduction.md
  - installation.md
  - usage.md
  - faq.md



# 3

Configure the Sidebar and Navigation

Create a _sidebar.md: In your docs folder, create a _sidebar.md file to define the structure of your documentation sidebar. Example:

- [Home](/)
- [Introduction](introduction.md)
- [Installation](installation.md)
- [Usage](usage.md)
- [FAQ](faq.md)

Configure Docsify: Modify the index.html to include the sidebar:

```
<script>
  window.$docsify = {
    loadSidebar: true,
    ...
  }
</script>
```


Docsify is highly customizable. You can modify the index.html to change settings, add themes, or enable features like full-text search, emoji support, and more. For example, to enable the search plugin, you can add the following script tag to your index.html:

```
<script>
    window.$docsify = {
        search: 'auto', // enables the search plugin
    }
</script>
<script src="//cdn.jsdelivr.net/npm/docsify/lib/plugins/search.min.js"></script>
```

To enable collapsibility of nested entries in the sidebar, do this:

```
  <script>
    window.$docsify = {
      
      alias: {
        '/.*/_sidebar.md': '/_sidebar.md',
      },
      subMaxLevel: 2, // when clicking on a document, the headings of the md are displayed in the sidebar for fast jumping
      		      // This is the depth to which they are shown (3 means h1, h2, h3 are shown)
      		      // With setting it to 0, this doesn't happen.
      sidebarDisplayLevel: 0, // At this depth the sidebar elements are collapsed at the start.
      			      // This being zero means all nested entries are collapsed from the start.
    }
  </script>
  
  
  <!-- Include the docsify-sidebar-collapse plugin -->
  <script src="//cdn.jsdelivr.net/npm/docsify-sidebar-collapse/dist/docsify-sidebar-collapse.min.js"></script>

  <!-- Optional: Include custom styles for the sidebar -->
  <!-- this gives you the nice > by the collapsible element.-->
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify-sidebar-collapse/dist/sidebar.min.css">
```


# 4
Serve Your Documentation Locally

To view your documentation site locally, navigate to the docs directory and start the local server:
```
docsify serve docs
```
Or if inside /docs:
```
docsify serve .
```
This command serves your documentation site at http://localhost:3000


# 5
 Publish Your Documentation

Push Changes to GitHub: Commit your changes and push them to your GitHub repository. \
Enable GitHub Pages: Go to the repository settings on GitHub, find the GitHub Pages section, and set the source to the master branch and /docs folder.



# 6: example initial index.html
```
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>My Project Documentation</title>
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/docsify/lib/themes/vue.css">
</head>
<body>
  <div id="app"></div>
  <script>
    window.$docsify = {
      name: 'My Project',
      repo: 'https://github.com/my/repo',
      loadSidebar: true
    }
  </script>
  <script src="//cdn.jsdelivr.net/npm/docsify/lib/docsify.min.js"></script>
</body>
</html>
```


# 7: make the sidebar navigation tree-like
Arrange your Markdown files in a hierarchical directory structure within the /docs folder. For example:

/docs
  - introduction.md
  - installation.md
  - usage.md
  - faq.md
  - guide/
    - guide-a.md
    - guide-b.md

Make _sidebar.md in your /docs directory. Structure it to reflect the hierarchy:

- [Home](/)
- [Introduction](introduction.md)
- [Installation](installation.md)
- [Usage](usage.md)
- [FAQ](faq.md)
- Guide
  - [Guide A](guide/guide-a.md)
  - [Guide B](guide/guide-b.md)




# 8 (just do by hand tbh - makes docs much more readable and better)
If you want to make the _sidebar.md automatically, you can:
https://github.com/docsifyjs/docsify/issues/1290
Place the Generator.zip file in your Docsify's directory which contains the .md files.
Extract it using:

unzip Generator.zip

Compile and Run the Java Program:
Compile the Java code:
javac -encoding UTF-8 Generator.java

Generate the sidebar:
java Generator build-sidebar .

This program will scan your directory structure and generate a _sidebar.md that reflects the hierarchical organization of your Markdown files.
