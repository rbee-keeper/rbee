# Image Generation Prompts for Hero Asides

**TEAM-XXX: AI image generation prompts for commercial app pages**

## Overview

Generate 5 images for pages that need visual asides. All images should:
- Be modern, professional, and on-brand
- Use rbee color palette (blues, purples, teals)
- Be optimized for web (PNG or WebP)
- Match specified dimensions

## Color Palette

**rbee Brand Colors:**
- Primary: `#3b82f6` (blue)
- Secondary: `#8b5cf6` (purple)
- Accent: `#06b6d4` (teal/cyan)
- Background: `#0f172a` (dark) / `#f8fafc` (light)
- Text: `#1e293b` (dark) / `#f1f5f9` (light)

## Image Specifications

| Page | Size | Aspect | Theme | Priority |
|------|------|--------|-------|----------|
| RhaiScriptingPage | 1024x1024 | Square | Code/scripting | HIGH |
| ResearchPage | 1024x1536 | Portrait | Academic | MEDIUM |
| EducationPage | 1024x1536 | Portrait | Learning | MEDIUM |
| StartupsPage | 1536x1024 | Landscape | Growth | LOW |
| CommunityPage | 1024x1024 | Square | Community | OPTIONAL |

---

## 1. RhaiScriptingPage (HIGH PRIORITY)

**Filename:** `rhai-scripting-hero.png`  
**Size:** 1024x1024 (square)  
**Theme:** Code, scripting, routing logic

### DALL-E 3 Prompt
```
Create a modern, abstract visualization of code and scripting logic. 
Show flowing code snippets in Rust/Rhai syntax with routing arrows and 
decision trees. Use a dark theme with neon blue (#3b82f6) and purple 
(#8b5cf6) accents. Include geometric shapes representing logic flow. 
Style: Modern tech illustration, clean and professional, isometric 
perspective. 1024x1024 pixels, high quality.
```

### Midjourney Prompt
```
abstract code visualization, Rhai scripting language, routing logic 
flow diagram, dark background, neon blue and purple accents, geometric 
shapes, modern tech aesthetic, isometric view, clean professional 
design, 1:1 aspect ratio --v 6 --style raw --ar 1:1
```

### Alternative Concept
```
Terminal window showing Rhai script with glowing syntax highlighting, 
surrounded by floating code blocks and routing arrows, dark theme with 
blue/purple gradient, modern developer aesthetic, 1024x1024
```

---

## 2. ResearchPage (MEDIUM PRIORITY)

**Filename:** `research-academic-hero.png`  
**Size:** 1024x1536 (portrait)  
**Theme:** Academic research, AI/ML, scholarly

### DALL-E 3 Prompt
```
Create an elegant academic research illustration. Show neural network 
diagrams, research papers, and data visualizations in a scholarly 
setting. Use a light, professional color palette with blue (#3b82f6) 
and purple (#8b5cf6) accents. Include abstract representations of AI 
models and research graphs. Style: Modern academic illustration, clean 
and trustworthy, vertical composition. 1024x1536 pixels, high quality.
```

### Midjourney Prompt
```
academic research visualization, neural networks, scholarly papers, 
data graphs, AI model diagrams, light professional background, blue 
and purple accents, modern academic aesthetic, clean trustworthy 
design, vertical composition --v 6 --style raw --ar 2:3
```

### Alternative Concept
```
Stack of research papers with floating AI/ML diagrams, neural network 
visualizations, and academic charts, light theme with professional 
color palette, scholarly atmosphere, 1024x1536
```

---

## 3. EducationPage (MEDIUM PRIORITY)

**Filename:** `education-learning-hero.png`  
**Size:** 1024x1536 (portrait)  
**Theme:** Learning, teaching, students, accessibility

### DALL-E 3 Prompt
```
Create a welcoming educational illustration showing diverse students 
learning together. Include laptops, books, and abstract learning 
symbols. Use warm, inviting colors with blue (#3b82f6) and teal 
(#06b6d4) accents. Show collaboration and knowledge sharing. Style: 
Modern educational illustration, inclusive and accessible, friendly 
atmosphere, vertical composition. 1024x1536 pixels, high quality.
```

### Midjourney Prompt
```
educational learning scene, diverse students collaborating, laptops 
and books, knowledge sharing symbols, warm inviting colors, blue and 
teal accents, modern educational aesthetic, inclusive friendly design, 
vertical composition --v 6 --style raw --ar 2:3
```

### Alternative Concept
```
Abstract representation of learning journey, books transforming into 
digital knowledge, students with laptops, collaborative atmosphere, 
warm colors with blue/teal accents, 1024x1536
```

---

## 4. StartupsPage (LOW PRIORITY)

**Filename:** `startups-growth-hero.png`  
**Size:** 1536x1024 (landscape)  
**Theme:** Startup growth, innovation, scaling

### DALL-E 3 Prompt
```
Create a dynamic startup growth visualization. Show upward trending 
graphs, rocket ship launch, and innovation symbols. Use energetic 
colors with blue (#3b82f6), purple (#8b5cf6), and teal (#06b6d4) 
accents. Include abstract representations of scaling and success. 
Style: Modern startup aesthetic, energetic and optimistic, horizontal 
composition. 1536x1024 pixels, high quality.
```

### Midjourney Prompt
```
startup growth visualization, upward trending graphs, rocket launch, 
innovation symbols, energetic colors, blue purple teal accents, modern 
startup aesthetic, optimistic dynamic design, horizontal composition 
--v 6 --style raw --ar 3:2
```

### Alternative Concept
```
Isometric view of startup office with growth charts, team 
collaboration, technology symbols, energetic atmosphere, blue/purple 
gradient, 1536x1024
```

---

## 5. CommunityPage (OPTIONAL)

**Filename:** `community-collaboration-hero.png`  
**Size:** 1024x1024 (square)  
**Theme:** Community, collaboration, open source

### DALL-E 3 Prompt
```
Create a vibrant community collaboration illustration. Show diverse 
people connecting through technology, GitHub stars, chat bubbles, and 
collaboration symbols. Use welcoming colors with blue (#3b82f6) and 
purple (#8b5cf6) accents. Include open source and community icons. 
Style: Modern community illustration, inclusive and welcoming, 
collaborative atmosphere. 1024x1024 pixels, high quality.
```

### Midjourney Prompt
```
community collaboration scene, diverse people connecting, technology 
network, GitHub stars, chat bubbles, welcoming colors, blue and purple 
accents, modern community aesthetic, inclusive collaborative design, 
1:1 aspect ratio --v 6 --style raw --ar 1:1
```

### Alternative Concept
```
Network of connected people with technology nodes, open source symbols, 
community icons, vibrant welcoming atmosphere, blue/purple palette, 
1024x1024
```

---

## Generation Tips

### For DALL-E 3:
1. Use detailed, descriptive prompts
2. Specify exact dimensions (1024x1024, etc.)
3. Request "high quality" explicitly
4. Mention "modern tech illustration" for consistency
5. Include color codes for brand consistency

### For Midjourney:
1. Use concise, comma-separated keywords
2. Add `--v 6` for latest version
3. Add `--style raw` for less stylization
4. Use `--ar` for aspect ratio (1:1, 2:3, 3:2)
5. Can add `--q 2` for higher quality

### For Stable Diffusion:
1. Use detailed positive prompts
2. Add negative prompts: "blurry, low quality, distorted"
3. Use ControlNet for consistent style
4. Set steps: 30-50 for quality
5. CFG scale: 7-9 for balance

## Post-Processing

After generation:

1. **Optimize:**
   ```bash
   # Convert to WebP (smaller file size)
   cwebp -q 85 input.png -o output.webp
   
   # Or optimize PNG
   optipng -o7 input.png
   ```

2. **Resize if needed:**
   ```bash
   # Using ImageMagick
   convert input.png -resize 1024x1024 output.png
   ```

3. **Save to:**
   ```
   frontend/apps/commercial/public/images/
   ```

## Usage in Props.tsx

After generating images:

```tsx
asideConfig: {
  variant: 'image',
  src: '/images/rhai-scripting-hero.png',
  alt: 'Rhai scripting visualization showing code flow and routing logic',
  width: 1024,
  height: 1024,
  title: 'User-Scriptable Routing',
  subtitle: 'Write custom logic in Rhai'
}
```

## Quality Checklist

- [ ] Correct dimensions (1024x1024, 1024x1536, or 1536x1024)
- [ ] Uses rbee brand colors
- [ ] Professional and modern aesthetic
- [ ] Clear and not cluttered
- [ ] Optimized file size (<500KB)
- [ ] Descriptive alt text prepared
- [ ] Saved to `/public/images/`
- [ ] Referenced in Props.tsx

## Backup: Use Existing Images

If AI generation isn't available, use existing images:

| Page | Existing Image | Notes |
|------|----------------|-------|
| RhaiScriptingPage | `features-rhai-routing.png` | Already exists! |
| ResearchPage | `og-academic.png` | May need resizing |
| EducationPage | `use-case-academic-hero-dark.png` | Academic theme |
| StartupsPage | Create or use generic | - |
| CommunityPage | `og-community.png` | May need resizing |

---

**Priority Order:**
1. RhaiScriptingPage (HIGH) - or use existing `features-rhai-routing.png`
2. ResearchPage (MEDIUM)
3. EducationPage (MEDIUM)
4. StartupsPage (LOW)
5. CommunityPage (OPTIONAL - can use stats aside instead)
