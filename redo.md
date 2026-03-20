# Website Redesign Plan

## Current State
- Jekyll-based blog site with posts in `_posts/` directory (8 blog posts)
- Landing page (`index.html`) displays blog post list
- About Me page exists at `/about/`
- Navigation header shows pages and resume link
- Blog-focused site title and description

## Desired State
- **Landing page**: About Me content (currently at `/about/`)
- **Lists page**: New blank template page
- **Thoughts page**: New page with sequential scrolling short posts
- **Remove**: All blog functionality and content

## Changes Required

### 1. Content Changes
- [ ] Convert `index.html` to display About Me content (make it the landing page)
- [ ] Remove or archive `_posts/` directory (8 blog posts to delete)
- [ ] Remove or archive `_drafts/` directory
- [ ] Remove old `about.md` file (content moved to index)
- [ ] Create new `lists.md` page with blank template
- [ ] Create new `thoughts.md` page with scrolling short posts layout

### 2. Configuration Changes
- [ ] Update `_config.yml`:
  - Change site title from "Dan Salo's Blog" to something non-blog
  - Update description to reflect new site purpose
  - Remove blog-related configurations if any

### 3. Layout Changes
- [ ] Keep `_layouts/default.html` (base layout)
- [ ] Keep `_layouts/page.html` (for Lists and other pages)
- [ ] Remove or keep `_layouts/post.html` (not needed, but won't hurt)
- [ ] Create new layout `_layouts/thoughts.html` for the Thoughts page with scrolling posts

### 4. Navigation Updates
- [ ] Update `_includes/header.html` navigation to show:
  - Home (About Me - the index)
  - Lists
  - Thoughts
  - Resume (keep existing)

### 5. Data Structure for Thoughts
- [ ] Create `_thoughts/` directory for thought posts
- [ ] Configure Jekyll to recognize thoughts collection in `_config.yml`
- [ ] Create sample thought posts to demonstrate functionality

### 6. Cleanup
- [ ] Remove Disqus comment integration (blog-specific)
- [ ] Update sitemap if needed
- [ ] Test all pages render correctly

## Implementation Order
1. Create planning doc (this file) ✓
2. Update site configuration
3. Create new pages (Lists, Thoughts)
4. Create Thoughts layout and collection
5. Convert index to About Me
6. Update navigation
7. Remove blog content
8. Clean up unused features

## Notes
- This is a Jekyll site, so changes follow Jekyll conventions
- Thoughts page will use a custom layout with sequential scrolling
- Lists page is intentionally blank template for future use
