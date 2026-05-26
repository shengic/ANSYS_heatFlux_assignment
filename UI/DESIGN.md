---
name: Kinetic Precision
colors:
  surface: '#f9f9fb'
  surface-dim: '#d9dadc'
  surface-bright: '#f9f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f5'
  surface-container: '#eeeef0'
  surface-container-high: '#e8e8ea'
  surface-container-highest: '#e2e2e4'
  on-surface: '#1a1c1d'
  on-surface-variant: '#46464a'
  inverse-surface: '#2f3132'
  inverse-on-surface: '#f0f0f2'
  outline: '#77767b'
  outline-variant: '#c7c6ca'
  surface-tint: '#5f5e60'
  primary: '#030304'
  on-primary: '#ffffff'
  primary-container: '#1d1d1f'
  on-primary-container: '#868587'
  inverse-primary: '#c8c6c8'
  secondary: '#5e5e63'
  on-secondary: '#ffffff'
  secondary-container: '#e0dfe4'
  on-secondary-container: '#626267'
  tertiary: '#000305'
  on-tertiary: '#ffffff'
  tertiary-container: '#00202c'
  on-tertiary-container: '#0090b8'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e4e2e4'
  primary-fixed-dim: '#c8c6c8'
  on-primary-fixed: '#1b1b1d'
  on-primary-fixed-variant: '#474649'
  secondary-fixed: '#e3e2e7'
  secondary-fixed-dim: '#c7c6cb'
  on-secondary-fixed: '#1a1b1f'
  on-secondary-fixed-variant: '#46464b'
  tertiary-fixed: '#bee9ff'
  tertiary-fixed-dim: '#68d3ff'
  on-tertiary-fixed: '#001f2a'
  on-tertiary-fixed-variant: '#004d64'
  background: '#f9f9fb'
  on-background: '#1a1c1d'
  surface-variant: '#e2e2e4'
  spectra-orange: '#ff6b35'
  ansys-blue: '#00c8ff'
  charcoal-surface: '#121212'
  metallic-silver: '#e8e8ed'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 30px
  body-lg:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: 0.01em
  label-sm-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
  mono-data:
    fontFamily: jetbrainsMono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  container-margin: 20px
  gutter: 12px
  card-padding: 16px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style

This design system bridges the gap between high-stakes engineering software and consumer-grade elegance. Inspired by modern hardware aesthetics, the visual direction prioritizes clarity, structural integrity, and premium finishes.

The style is a blend of **Minimalism** and **Glassmorphism**. It utilizes expansive white space (or deep charcoal backgrounds) to reduce cognitive load while employing translucent layers to maintain context in high-density data environments. The aesthetic response should feel "calmly powerful"—an interface that feels as responsive and precise as the simulations it maps.

## Colors

The palette is rooted in a monochromatic foundation of "Ebonies" and "Silvers" to reflect a high-tech hardware feel. 

- **Primary & Neutrals:** Use #1d1d1f for text and primary branding to mirror premium industrial design. Use the neutral #f5f5f7 for large background surfaces to keep the mobile interface airy.
- **Accents:** The **ANSYS Blue** is the primary functional accent for active states, primary actions, and successful simulation results. The **SPECTRA Orange** is reserved for warnings, critical data points, and attention-required status indicators.
- **Data Visualization:** When mapping complex VBA outputs, use varying opacities of the accent colors to represent density and heat, ensuring they pop against the charcoal or white backgrounds.

## Typography

The system uses **Inter** for its systematic, neutral character and excellent legibility at small sizes. To achieve the "Apple-esque" hierarchy, use a high contrast between weights: pairing Bold (700) or Semi-Bold (600) headlines with Regular (400) body text.

For the VBA simulation data specifically, a secondary monospaced font (**JetBrains Mono**) is introduced for technical values, coordinates, and code snippets to ensure numerical alignment and a "developer-tool" aesthetic. All labels should be tight and clear, using uppercase tracking for secondary metadata.

## Layout & Spacing

The layout follows a **Fluid Grid** model optimized for high-density information. On mobile, the system uses a 4-column structure with 20px side margins.

- **Information Density:** Given the complexity of VBA mapping, use a tight 4px baseline grid. Elements should be grouped into logical "clusters" using padding rather than heavy lines to maintain a clean look.
- **Reflow:** On tablet/landscape view, the layout expands to a 12-column grid, allowing for a persistent sidebar containing simulation parameters while the main canvas displays the map.
- **Safe Areas:** Ensure all interactive elements adhere to a 44px minimum touch target, even if the visual representation (like a small data chip) is smaller.

## Elevation & Depth

This system uses **Glassmorphism** and **Tonal Layers** to create a sense of organized depth without visual clutter.

1.  **The Canvas (Level 0):** The base background layer, typically Crisp White (#FFFFFF) or Deep Charcoal (#121212).
2.  **Floating Cards (Level 1):** Elements sit on a surface with a very subtle 1px border (#E8E8ED) and a soft, diffused ambient shadow (10% opacity, 20px blur).
3.  **Glass Overlays (Level 2):** Used for navigation bars and modal sheets. These employ a backdrop filter (blur: 20px) and a semi-transparent background (rgba(255, 255, 255, 0.7)).
4.  **Interactive States:** On press, cards should slightly "sink" (reduce shadow and scale to 0.98) to provide tactile feedback typical of a sophisticated engineering tool.

## Shapes

The shape language is "Softly Geometric." A consistent **roundedness level of 2** (0.5rem base) is applied to all standard components.

- **Outer Containers:** Large dashboard cards and modal sheets should use `rounded-xl` (1.5rem / 24px) to create a friendly, approachable frame for the technical data inside.
- **Inner Elements:** Buttons and input fields use `rounded-lg` (1rem / 16px).
- **Status Chips:** Small data indicators should be fully pill-shaped to differentiate them from interactive buttons.

## Components

- **Buttons:** Primary buttons are solid Charcoal (#1d1d1f) with white text. Secondary buttons use a "Ghost" style with a 1px metallic gray border.
- **Data Chips:** Compact, pill-shaped indicators. Use **ANSYS Blue** for "active" and **SPECTRA Orange** for "alert." Use a light tint of the color for the background and the full saturation for the text/icon.
- **Simulation Cards:** These are the primary vessel for information. They must feature a header with a bold weight title, a JetBrains Mono data readout, and a subtle 0.5px separator line.
- **Glass Navigation:** A bottom bar or top header with a heavy backdrop blur, ensuring content remains visible but blurred as it scrolls beneath.
- **Inputs:** Minimalist text fields with a bottom-only border that transforms into a full-focus ring in ANSYS Blue when active.
- **Icons:** Use thin-stroke (1.5pt) linear icons. Avoid filled icons unless they represent an active toggle state.