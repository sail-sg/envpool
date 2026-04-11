/*
  SDL_ttf:  A companion library to SDL for working with TrueType (tm) fonts
  Copyright (C) 2001-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#include "SDL2/SDL.h"
#include "SDL2/SDL_cpuinfo.h"
#include "SDL2/SDL_endian.h"

#include "SDL_ttf.h"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H
#include FT_STROKER_H
#include FT_GLYPH_H
#include FT_TRUETYPE_IDS_H
#include FT_IMAGE_H

/* Enable rendering with color
 * Freetype may need to be compiled with FT_CONFIG_OPTION_USE_PNG */
#if defined(FT_HAS_COLOR)
#  define TTF_USE_COLOR 1
#else
#  define TTF_USE_COLOR 0
#endif

/* Enable Signed Distance Field rendering (requires latest FreeType version) */
#if defined(FT_RASTER_FLAG_SDF)
#  define TTF_USE_SDF 1
#else
#  define TTF_USE_SDF 0
#endif

#if TTF_USE_SDF
#include FT_MODULE_H
#endif

/* Enable HarfBuzz for Complex text rendering
 * Freetype may need to be compiled with FT_CONFIG_OPTION_USE_HARFBUZZ */
#ifndef TTF_USE_HARFBUZZ
#  define TTF_USE_HARFBUZZ 0
#endif

#if defined(SDL_BUILD_MAJOR_VERSION) && defined(SDL_COMPILE_TIME_ASSERT)
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MAJOR_VERSION,
                        SDL_TTF_MAJOR_VERSION == SDL_BUILD_MAJOR_VERSION);
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MINOR_VERSION,
                        SDL_TTF_MINOR_VERSION == SDL_BUILD_MINOR_VERSION);
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MICRO_VERSION,
                        SDL_TTF_PATCHLEVEL == SDL_BUILD_MICRO_VERSION);
#endif

#if defined(SDL_COMPILE_TIME_ASSERT)
SDL_COMPILE_TIME_ASSERT(SDL_TTF_MAJOR_VERSION_min, SDL_TTF_MAJOR_VERSION >= 0);
/* Limited only by the need to fit in SDL_version */
SDL_COMPILE_TIME_ASSERT(SDL_TTF_MAJOR_VERSION_max, SDL_TTF_MAJOR_VERSION <= 255);

SDL_COMPILE_TIME_ASSERT(SDL_TTF_MINOR_VERSION_min, SDL_TTF_MINOR_VERSION >= 0);
/* Limited only by the need to fit in SDL_version */
SDL_COMPILE_TIME_ASSERT(SDL_TTF_MINOR_VERSION_max, SDL_TTF_MINOR_VERSION <= 255);

SDL_COMPILE_TIME_ASSERT(SDL_TTF_PATCHLEVEL_min, SDL_TTF_PATCHLEVEL >= 0);
/* Limited by its encoding in SDL_VERSIONNUM and in the ABI versions */
SDL_COMPILE_TIME_ASSERT(SDL_TTF_PATCHLEVEL_max, SDL_TTF_PATCHLEVEL <= 99);
#endif

#if TTF_USE_HARFBUZZ
#include <hb.h>
#include <hb-ft.h>

/* Default configuration */
static hb_direction_t g_hb_direction = HB_DIRECTION_LTR;
static hb_script_t    g_hb_script = HB_SCRIPT_UNKNOWN;
#endif

/* Harfbuzz */
int TTF_SetDirection(int direction) /* hb_direction_t */
{
#if TTF_USE_HARFBUZZ
    g_hb_direction = direction;
    return 0;
#else
    (void) direction;
    return -1;
#endif
}

int TTF_SetScript(int script) /* hb_script_t */
{
#if TTF_USE_HARFBUZZ
    g_hb_script = script;
    return 0;
#else
    (void) script;
    return -1;
#endif
}

/* Round glyph to 16 bytes width and use SSE2 instructions */
#if defined(__SSE2__)
#  define HAVE_SSE2_INTRINSICS 1
#endif

/* Round glyph width to 16 bytes use NEON instructions */
#if 0 /*defined(__ARM_NEON)*/
#  define HAVE_NEON_INTRINSICS 1
#endif

/* Round glyph width to 8 bytes */
#define HAVE_BLIT_GLYPH_64

/* Android armeabi-v7a doesn't like int64 (Maybe all other __ARM_ARCH < 7 ?),
 * un-activate it, especially if NEON isn't detected */
#if defined(__ARM_ARCH)
#  if __ARM_ARCH < 8
#    if defined(HAVE_BLIT_GLYPH_64)
#      undef HAVE_BLIT_GLYPH_64
#    endif
#  endif
#endif

/* Default: round glyph width to 4 bytes to copy them faster */
#define HAVE_BLIT_GLYPH_32

/* Use Duff's device to unroll loops */
//#define USE_DUFFS_LOOP

#if defined(HAVE_SSE2_INTRINSICS)
static SDL_INLINE int hasSSE2(void)
{
    static int val = -1;
    if (val != -1) {
        return val;
    }
    val = SDL_HasSSE2();
    return val;
}
#endif

#if defined(HAVE_NEON_INTRINSICS)
static SDL_INLINE int hasNEON(void)
{
    static int val = -1;
    if (val != -1) {
        return val;
    }
    val = SDL_HasNEON();
    return val;
}
#endif

/* FIXME: Right now we assume the gray-scale renderer Freetype is using
   supports 256 shades of gray, but we should instead key off of num_grays
   in the result FT_Bitmap after the FT_Render_Glyph() call. */
#define NUM_GRAYS       256

/* x offset = cos(((90.0-12)/360) * 2 * M_PI), or 12 degree angle */
/* same value as in FT_GlyphSlot_Oblique, fixed point 16.16 */
#define GLYPH_ITALICS  0x0366AL

/* Handy routines for converting from fixed point 26.6 */
#define FT_FLOOR(X) (int)(((X) & -64) / 64)
#define FT_CEIL(X)  FT_FLOOR((X) + 63)

/* Handy routine for converting to fixed point 26.6 */
#define F26Dot6(X)  ((X) << 6)

/* Faster divide by 255, with same result
 * in range [0; 255]:  (x + 1   + (x >> 8)) >> 8
 * in range [-255; 0]: (x + 255 + (x >> 8)) >> 8 */
#define DIVIDE_BY_255_SIGNED(x, sign_val)  (((x) + (sign_val) + ((x)>>8)) >> 8)

/* When x positive */
#define DIVIDE_BY_255(x)    DIVIDE_BY_255_SIGNED(x, 1)


#define CACHED_METRICS  0x20

#define CACHED_BITMAP   0x01
#define CACHED_PIXMAP   0x02
#define CACHED_COLOR    0x04
#define CACHED_LCD      0x08
#define CACHED_SUBPIX   0x10


typedef struct {
    unsigned char *buffer; /* aligned */
    int            left;
    int            top;
    int            width;
    int            rows;
    int            pitch;
    int            is_color;
} TTF_Image;

/* Cached glyph information */
typedef struct cached_glyph {
    int stored;
    FT_UInt index;
    TTF_Image bitmap;
    TTF_Image pixmap;
    int sz_left;
    int sz_top;
    int sz_width;
    int sz_rows;
    int advance;
    union {
        /* TTF_HINTING_LIGHT_SUBPIXEL (only pixmap) */
        struct {
            int lsb_minus_rsb;
            int translation;
        } subpixel;
        /* Other hinting */
        struct {
            int rsb_delta;
            int lsb_delta;
        } kerning_smart;
    };
} c_glyph;

/* Internal buffer to store positions computed by TTF_Size_Internal()
 * for rendered string by Render_Line() */
typedef struct PosBuf {
    FT_UInt index;
    int x;
    int y;
} PosBuf_t;

/* The structure used to hold internal font information */
struct TTF_Font {
    /* Freetype2 maintains all sorts of useful info itself */
    FT_Face face;

    /* We'll cache these ourselves */
    int height;
    int ascent;
    int descent;
    int lineskip;

    /* The font style */
    int style;
    int outline_val;

    /* Whether kerning is desired */
    int allow_kerning;
#if !TTF_USE_HARFBUZZ
    int use_kerning;
#endif

    /* Extra width in glyph bounds for text styles */
    int glyph_overhang;

    /* Information in the font for underlining */
    int line_thickness;
    int underline_top_row;
    int strikethrough_top_row;

    /* Cache for style-transformed glyphs */
    c_glyph cache[256];
    FT_UInt cache_index[128];

    /* We are responsible for closing the font stream */
    SDL_RWops *src;
    int freesrc;
    FT_Open_Args args;

    /* Internal buffer to store positions computed by TTF_Size_Internal()
     * for rendered string by Render_Line() */
    PosBuf_t *pos_buf;
    Uint32 pos_len;
    Uint32 pos_max;

    /* Hinting modes */
    int ft_load_target;
    int render_subpixel;
#if TTF_USE_HARFBUZZ
    hb_font_t *hb_font;
    /* If HB_SCRIPT_INVALID, use global default script */
    hb_script_t hb_script;
    /* If HB_DIRECTION_INVALID, use global default direction */
    hb_direction_t hb_direction;
#endif
    int render_sdf;

    /* Extra layout setting for wrapped text */
    int horizontal_align;
};

/* Tell if SDL_ttf has to handle the style */
#define TTF_HANDLE_STYLE_BOLD(font)          ((font)->style & TTF_STYLE_BOLD)
#define TTF_HANDLE_STYLE_ITALIC(font)        ((font)->style & TTF_STYLE_ITALIC)
#define TTF_HANDLE_STYLE_UNDERLINE(font)     ((font)->style & TTF_STYLE_UNDERLINE)
#define TTF_HANDLE_STYLE_STRIKETHROUGH(font) ((font)->style & TTF_STYLE_STRIKETHROUGH)

/* Font styles that does not impact glyph drawing */
#define TTF_STYLE_NO_GLYPH_CHANGE   (TTF_STYLE_UNDERLINE | TTF_STYLE_STRIKETHROUGH)

/* The FreeType font engine/library */
static FT_Library library = NULL;
static int TTF_initialized = 0;
static SDL_bool TTF_byteswapped = SDL_FALSE;

#define TTF_CHECK_INITIALIZED(errval)                   \
    if (!TTF_initialized) {                             \
        TTF_SetError("Library not initialized");        \
        return errval;                                  \
    }

#define TTF_CHECK_POINTER(p, errval)                    \
    if (!(p)) {                                         \
        TTF_SetError("Passed a NULL pointer");          \
        return errval;                                  \
    }

typedef enum {
    RENDER_SOLID = 0,
    RENDER_SHADED,
    RENDER_BLENDED,
    RENDER_LCD
} render_mode_t;

typedef enum {
    STR_UTF8 = 0,
    STR_TEXT,
    STR_UNICODE
} str_type_t;

static int TTF_initFontMetrics(TTF_Font *font);

static int TTF_Size_Internal(TTF_Font *font, const char *text, str_type_t str_type,
        int *w, int *h, int *xstart, int *ystart, int measure_width, int *extent, int *count);

#define NO_MEASUREMENT  \
        0, NULL, NULL


static SDL_Surface* TTF_Render_Internal(TTF_Font *font, const char *text, str_type_t str_type,
        SDL_Color fg, SDL_Color bg, render_mode_t render_mode);

static SDL_Surface* TTF_Render_Wrapped_Internal(TTF_Font *font, const char *text, str_type_t str_type,
        SDL_Color fg, SDL_Color bg, Uint32 wrapLength, render_mode_t render_mode);

static SDL_INLINE int Find_GlyphByIndex(TTF_Font *font, FT_UInt idx,
        int want_bitmap, int want_pixmap, int want_color, int want_lcd, int want_subpixel,
        int translation, c_glyph **out_glyph, TTF_Image **out_image);

static void Flush_Cache(TTF_Font *font);

#if defined(USE_DUFFS_LOOP)

/* 4-times unrolled loop */
#define DUFFS_LOOP4(pixel_copy_increment, width)                        \
{ int n = (width+3)/4;                                                  \
    switch (width & 3) {                                                \
    case 0: do {    pixel_copy_increment;   /* fallthrough */           \
    case 3:     pixel_copy_increment;       /* fallthrough */           \
    case 2:     pixel_copy_increment;       /* fallthrough */           \
    case 1:     pixel_copy_increment;       /* fallthrough */           \
        } while (--n > 0);                                              \
    }                                                                   \
}

#else

/* Don't use Duff's device to unroll loops */
#define DUFFS_LOOP(pixel_copy_increment, width)                         \
{ int n;                                                                \
    for ( n=width; n > 0; --n ) {                                       \
        pixel_copy_increment;                                           \
    }                                                                   \
}

#define DUFFS_LOOP4(pixel_copy_increment, width)                        \
    DUFFS_LOOP(pixel_copy_increment, width)

#endif


/* Blend colored glyphs */
static SDL_INLINE void BG_Blended_Color(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const Uint32 *src   = (Uint32 *)image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    if (fg_alpha == 0) { /* SDL_ALPHA_OPAQUE */
        while (height--) {
            /* *INDENT-OFF* */
            DUFFS_LOOP4(
                *dst++ = *src++;
            , width);
            /* *INDENT-ON* */
            src = (const Uint32 *)((const Uint8 *)src + srcskip);
            dst = (Uint32 *)((Uint8 *)dst + dstskip);
        }
    } else {
        Uint32 alpha;
        Uint32 tmp;

        while (height--) {
            /* *INDENT-OFF* */
            DUFFS_LOOP4(
                    /* prevent misaligned load: tmp = *src++; */
                    /* eventually, we can expect the compiler to replace the memcpy call with something optimized */
                    memcpy(&tmp, src++, sizeof(tmp)); /* This should NOT be SDL_memcpy */
                    alpha = tmp >> 24;
                    tmp &= ~0xFF000000;
                    alpha = fg_alpha * alpha;
                    alpha =  DIVIDE_BY_255(alpha) << 24;
                    *dst++ = tmp | alpha
                    , width);
            /* *INDENT-ON* */
            src = (const Uint32 *)((const Uint8 *)src + srcskip);
            dst = (Uint32 *)((Uint8 *)dst + dstskip);
        }
    }
}

/* Blend with LCD rendering */
static SDL_INLINE void BG_Blended_LCD(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, SDL_Color *fg)
{
    const Uint32 *src   = (Uint32 *)image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    Uint32 tmp, bg;
    Uint32 r, g, b;
    Uint8 fg_r, fg_g, fg_b;
    Uint8 bg_r, bg_g, bg_b;
    Uint32 bg_a;

    fg_r = fg->r;
    fg_g = fg->g;
    fg_b = fg->b;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
                /* prevent misaligned load: tmp = *src++; */
                memcpy(&tmp, src++, sizeof(tmp)); /* This should NOT be SDL_memcpy */

                if (tmp) {
                    bg = *dst;

                    bg_a = bg & 0xff000000;
                    bg_r = (bg >> 16) & 0xff;
                    bg_g = (bg >> 8) & 0xff;
                    bg_b = (bg >> 0) & 0xff;

                    r = (tmp >> 16) & 0xff;
                    g = (tmp >> 8) & 0xff;
                    b = (tmp >> 0) & 0xff;

                    r = fg_r * r + bg_r * (255 - r) + 127;
                    r = DIVIDE_BY_255(r);

                    g = fg_g * g + bg_g * (255 - g) + 127;
                    g = DIVIDE_BY_255(g);

                    b = fg_b * b + bg_b * (255 - b) + 127;
                    b = DIVIDE_BY_255(b);

                    r <<= 16;
                    g <<= 8;
                    b <<= 0;

                    *dst = r | g | b | bg_a;
                }
                dst++;

                , width);
        /* *INDENT-ON* */
        src = (const Uint32 *)((const Uint8 *)src + srcskip);
        dst = (Uint32 *)((Uint8 *)dst + dstskip);
    }

}

#if TTF_USE_SDF

/* Blended Opaque SDF */
static SDL_INLINE void BG_Blended_Opaque_SDF(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    Uint32 s;
    Uint32 d;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            d = *dst;
            s = ((Uint32)*src++) << 24;
            if (s > d) {
                *dst = s;
            }
            dst++;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

/* Blended non-opaque SDF */
static SDL_INLINE void BG_Blended_SDF(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    Uint32 s;
    Uint32 d;

    Uint32 tmp;
    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            d = *dst;
            tmp = fg_alpha * (*src++);
            s = DIVIDE_BY_255(tmp) << 24;
            if (s > d) {
                *dst = s;
            }
            dst++;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

#endif /* TTF_USE_SDF */

/* Blended Opaque */
static SDL_INLINE void BG_Blended_Opaque(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            *dst++ |= ((Uint32)*src++) << 24;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

/* Blended non-opaque */
static SDL_INLINE void BG_Blended(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    Uint32 tmp;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            tmp     = fg_alpha * (*src++);
            *dst++ |= DIVIDE_BY_255(tmp) << 24;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

#if defined(HAVE_BLIT_GLYPH_32) || defined(HAVE_BLIT_GLYPH_64)
static SDL_INLINE void BG_Blended_Opaque_32(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width / 4;
    Uint32       height = image->rows;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            *dst++ |= ((Uint32)*src++) << 24;
            *dst++ |= ((Uint32)*src++) << 24;
            *dst++ |= ((Uint32)*src++) << 24;
            *dst++ |= ((Uint32)*src++) << 24;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

static SDL_INLINE void BG_Blended_32(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const Uint8 *src    = image->buffer;
    Uint32      *dst    = destination;
    Uint32       width  = image->width / 4;
    Uint32       height = image->rows;

    Uint32 tmp0, tmp1, tmp2, tmp3;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            tmp0    = fg_alpha * (*src++);
            tmp1    = fg_alpha * (*src++);
            tmp2    = fg_alpha * (*src++);
            tmp3    = fg_alpha * (*src++);
            *dst++ |= DIVIDE_BY_255(tmp0) << 24;
            *dst++ |= DIVIDE_BY_255(tmp1) << 24;
            *dst++ |= DIVIDE_BY_255(tmp2) << 24;
            *dst++ |= DIVIDE_BY_255(tmp3) << 24;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst  = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}
#endif

#if defined(HAVE_SSE2_INTRINSICS)
/* Apply: alpha_table[i] = i << 24; */
static SDL_INLINE void BG_Blended_Opaque_SSE(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const __m128i *src    = (__m128i *)image->buffer;
    __m128i       *dst    = (__m128i *)destination;
    Uint32         width  = image->width / 16;
    Uint32         height = image->rows;

    __m128i s, s0, s1, s2, s3, d0, d1, d2, d3, r0, r1, r2, r3, L, H;
    const __m128i zero  = _mm_setzero_si128();

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            /* Read 16 Uint8 at once and put into 4 __m128i */
            s  = _mm_loadu_si128(src);          // load unaligned
            d0 = _mm_load_si128(dst);           // load
            d1 = _mm_load_si128(dst + 1);       // load
            d2 = _mm_load_si128(dst + 2);       // load
            d3 = _mm_load_si128(dst + 3);       // load

            L  = _mm_unpacklo_epi8(zero, s);
            H  = _mm_unpackhi_epi8(zero, s);

            s0 = _mm_unpacklo_epi8(zero, L);
            s1 = _mm_unpackhi_epi8(zero, L);
            s2 = _mm_unpacklo_epi8(zero, H);
            s3 = _mm_unpackhi_epi8(zero, H);
                                                // already shifted by 24
            r0 = _mm_or_si128(d0, s0);          // or
            r1 = _mm_or_si128(d1, s1);          // or
            r2 = _mm_or_si128(d2, s2);          // or
            r3 = _mm_or_si128(d3, s3);          // or

            _mm_store_si128(dst, r0);           // store
            _mm_store_si128(dst + 1, r1);       // store
            _mm_store_si128(dst + 2, r2);       // store
            _mm_store_si128(dst + 3, r3);       // store

            dst += 4;
            src += 1;
        , width);
        /* *INDENT-ON* */
        src = (const __m128i *)((const Uint8 *)src + srcskip);
        dst = (__m128i *)((Uint8 *)dst + dstskip);
    }
}

static SDL_INLINE void BG_Blended_SSE(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const __m128i *src    = (__m128i *)image->buffer;
    __m128i       *dst    = (__m128i *)destination;
    Uint32         width  = image->width / 16;
    Uint32         height = image->rows;

    const __m128i alpha = _mm_set1_epi16(fg_alpha);
    const __m128i one   = _mm_set1_epi16(1);
    const __m128i zero  = _mm_setzero_si128();
    __m128i s, s0, s1, s2, s3, d0, d1, d2, d3, r0, r1, r2, r3, L, H, Ls8, Hs8;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            /* Read 16 Uint8 at once and put into 4 __m128i */
            s  = _mm_loadu_si128(src);          // load unaligned
            d0 = _mm_load_si128(dst);           // load
            d1 = _mm_load_si128(dst + 1);       // load
            d2 = _mm_load_si128(dst + 2);       // load
            d3 = _mm_load_si128(dst + 3);       // load

            L  = _mm_unpacklo_epi8(s, zero);    // interleave, no shifting
            H  = _mm_unpackhi_epi8(s, zero);    // enough room to multiply

            /* Apply: alpha_table[i] = ((i * fg.a / 255) << 24; */
            /* Divide by 255 is done as:    (x + 1 + (x >> 8)) >> 8 */

            L  = _mm_mullo_epi16(L, alpha);     // x := i * fg.a
            H  = _mm_mullo_epi16(H, alpha);

            Ls8 = _mm_srli_epi16(L, 8);         // x >> 8
            Hs8 = _mm_srli_epi16(H, 8);
            L = _mm_add_epi16(L, one);          // x + 1
            H = _mm_add_epi16(H, one);
            L = _mm_add_epi16(L, Ls8);          // x + 1 + (x >> 8)
            H = _mm_add_epi16(H, Hs8);
            L = _mm_srli_epi16(L, 8);           // ((x + 1 + (x >> 8)) >> 8
            H = _mm_srli_epi16(H, 8);

            L = _mm_slli_epi16(L, 8);           // shift << 8, so we're prepared
            H = _mm_slli_epi16(H, 8);           // to have final format << 24

            s0 = _mm_unpacklo_epi8(zero, L);
            s1 = _mm_unpackhi_epi8(zero, L);
            s2 = _mm_unpacklo_epi8(zero, H);
            s3 = _mm_unpackhi_epi8(zero, H);
                                                // already shifted by 24

            r0 = _mm_or_si128(d0, s0);          // or
            r1 = _mm_or_si128(d1, s1);          // or
            r2 = _mm_or_si128(d2, s2);          // or
            r3 = _mm_or_si128(d3, s3);          // or

            _mm_store_si128(dst, r0);           // store
            _mm_store_si128(dst + 1, r1);       // store
            _mm_store_si128(dst + 2, r2);       // store
            _mm_store_si128(dst + 3, r3);       // store

            dst += 4;
            src += 1;
        , width);
        /* *INDENT-ON* */
        src = (const __m128i *)((const Uint8 *)src + srcskip);
        dst = (__m128i *)((Uint8 *)dst + dstskip);
    }
}
#endif

#if defined(HAVE_NEON_INTRINSICS)
/* Apply: alpha_table[i] = i << 24; */
static SDL_INLINE void BG_Blended_Opaque_NEON(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint32 *src    = (Uint32 *)image->buffer;
    Uint32       *dst    = destination;
    Uint32        width  = image->width / 16;
    Uint32        height = image->rows;

    uint32x4_t s, d0, d1, d2, d3, r0, r1, r2, r3;
    uint8x16x2_t sx, sx01, sx23;
    const uint8x16_t zero = vdupq_n_u8(0);

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            /* Read 4 Uint32 and put 16 Uint8 into uint32x4x2_t (uint8x16x2_t)
             * takes advantage of vzipq_u8 which produces two lanes */

            s   = vld1q_u32(src);                           // load
            d0  = vld1q_u32(dst);                           // load
            d1  = vld1q_u32(dst + 4);                       // load
            d2  = vld1q_u32(dst + 8);                       // load
            d3  = vld1q_u32(dst + 12);                      // load

            sx   = vzipq_u8(zero, (uint8x16_t)s);           // interleave
            sx01 = vzipq_u8(zero, sx.val[0]);               // interleave
            sx23 = vzipq_u8(zero, sx.val[1]);               // interleave
                                                            // already shifted by 24
            r0  = vorrq_u32(d0, (uint32x4_t)sx01.val[0]);   // or
            r1  = vorrq_u32(d1, (uint32x4_t)sx01.val[1]);   // or
            r2  = vorrq_u32(d2, (uint32x4_t)sx23.val[0]);   // or
            r3  = vorrq_u32(d3, (uint32x4_t)sx23.val[1]);   // or

            vst1q_u32(dst, r0);                             // store
            vst1q_u32(dst + 4, r1);                         // store
            vst1q_u32(dst + 8, r2);                         // store
            vst1q_u32(dst + 12, r3);                        // store

            dst += 16;
            src += 4;
        , width);
        /* *INDENT-ON* */
        src = (const Uint32 *)((const Uint8 *)src + srcskip);
        dst = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}

/* Non-opaque, computes alpha blending on the fly */
static SDL_INLINE void BG_Blended_NEON(const TTF_Image *image, Uint32 *destination, Sint32 srcskip, Uint32 dstskip, Uint8 fg_alpha)
{
    const Uint32 *src    = (Uint32 *)image->buffer;
    Uint32       *dst    = destination;
    Uint32        width  = image->width / 16;
    Uint32        height = image->rows;

    uint32x4_t s, d0, d1, d2, d3, r0, r1, r2, r3;
    uint16x8_t Ls8, Hs8;
    uint8x16x2_t sx, sx01, sx23;
    uint16x8x2_t sxm;

    const uint16x8_t alpha = vmovq_n_u16(fg_alpha);
    const uint16x8_t one   = vmovq_n_u16(1);
    const uint8x16_t zero  = vdupq_n_u8(0);

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            /* Read 4 Uint32 and put 16 Uint8 into uint32x4x2_t (uint8x16x2_t)
             * takes advantage of vzipq_u8 which produces two lanes */

            s  = vld1q_u32(src);                                    // load
            d0 = vld1q_u32(dst);                                    // load
            d1 = vld1q_u32(dst + 4);                                // load
            d2 = vld1q_u32(dst + 8);                                // load
            d3 = vld1q_u32(dst + 12);                               // load

            sx = vzipq_u8((uint8x16_t)s, zero);                     // interleave, no shifting
                                                                    // enough room to multiply

            /* Apply: alpha_table[i] = ((i * fg.a / 255) << 24; */
            /* Divide by 255 is done as:    (x + 1 + (x >> 8)) >> 8 */

            sxm.val[0] = vmulq_u16((uint16x8_t)sx.val[0], alpha);   // x := i * fg.a
            sxm.val[1] = vmulq_u16((uint16x8_t)sx.val[1], alpha);

            Ls8 = vshrq_n_u16(sxm.val[0], 8);                       // x >> 8
            Hs8 = vshrq_n_u16(sxm.val[1], 8);

            sxm.val[0] = vaddq_u16(sxm.val[0], one);                // x + 1
            sxm.val[1] = vaddq_u16(sxm.val[1], one);

            sxm.val[0] = vaddq_u16(sxm.val[0], Ls8);                // x + 1 + (x >> 8)
            sxm.val[1] = vaddq_u16(sxm.val[1], Hs8);

            sxm.val[0] = vshrq_n_u16(sxm.val[0], 8);                // ((x + 1 + (x >> 8)) >> 8
            sxm.val[1] = vshrq_n_u16(sxm.val[1], 8);

            sxm.val[0] = vshlq_n_u16(sxm.val[0], 8);                // shift << 8, so we're prepared
            sxm.val[1] = vshlq_n_u16(sxm.val[1], 8);                // to have final format << 24

            sx01 = vzipq_u8(zero, (uint8x16_t)sxm.val[0]);          // interleave
            sx23 = vzipq_u8(zero, (uint8x16_t)sxm.val[1]);          // interleave
                                                                    // already shifted by 24

            r0  = vorrq_u32(d0, (uint32x4_t)sx01.val[0]);           // or
            r1  = vorrq_u32(d1, (uint32x4_t)sx01.val[1]);           // or
            r2  = vorrq_u32(d2, (uint32x4_t)sx23.val[0]);           // or
            r3  = vorrq_u32(d3, (uint32x4_t)sx23.val[1]);           // or

            vst1q_u32(dst, r0);                                     // store
            vst1q_u32(dst + 4, r1);                                 // store
            vst1q_u32(dst + 8, r2);                                 // store
            vst1q_u32(dst + 12, r3);                                // store

            dst += 16;
            src += 4;
        , width);
        /* *INDENT-ON* */
        src = (const Uint32 *)((const Uint8 *)src + srcskip);
        dst = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}
#endif

static SDL_INLINE void BG(const TTF_Image *image, Uint8 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint8 *src    = image->buffer;
    Uint8       *dst    = destination;
    Uint32       width  = image->width;
    Uint32       height = image->rows;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            *dst++ |= *src++;
        , width);
        /* *INDENT-ON* */
        src += srcskip;
        dst += dstskip;
    }
}

#if defined(HAVE_BLIT_GLYPH_64)
static SDL_INLINE void BG_64(const TTF_Image *image, Uint8 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint64 *src    = (Uint64 *)image->buffer;
    Uint64       *dst    = (Uint64 *)destination;
    Uint32        width  = image->width / 8;
    Uint32        height = image->rows;
    Uint64        tmp;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
              /* prevent misaligned load: *dst++ |= *src++; */
              memcpy(&tmp, src++, sizeof(tmp)); /* This should NOT be SDL_memcpy */
              *dst++ |= tmp;
        , width);
        /* *INDENT-ON* */
        src = (const Uint64 *)((const Uint8 *)src + srcskip);
        dst = (Uint64 *)((Uint8 *)dst + dstskip);
    }
}
#elif defined(HAVE_BLIT_GLYPH_32)
static SDL_INLINE void BG_32(const TTF_Image *image, Uint8 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint32 *src    = (Uint32 *)image->buffer;
    Uint32       *dst    = (Uint32 *)destination;
    Uint32        width  = image->width / 4;
    Uint32        height = image->rows;
    Uint32        tmp;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            /* prevent misaligned load: *dst++ |= *src++; */
            memcpy(&tmp, src++, sizeof(tmp)); /* This should NOT be SDL_memcpy */
            *dst++ |= tmp;
        , width);
        /* *INDENT-ON* */
        src = (const Uint32 *)((const Uint8 *)src + srcskip);
        dst = (Uint32 *)((Uint8 *)dst + dstskip);
    }
}
#endif

#if defined(HAVE_SSE2_INTRINSICS)
static SDL_INLINE void BG_SSE(const TTF_Image *image, Uint8 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const __m128i *src    = (__m128i *)image->buffer;
    __m128i       *dst    = (__m128i *)destination;
    Uint32         width  = image->width / 16;
    Uint32         height = image->rows;

    __m128i s, d, r;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            s = _mm_loadu_si128(src);   // load unaligned
            d = _mm_load_si128(dst);    // load
            r = _mm_or_si128(d, s);     // or
            _mm_store_si128(dst, r);    // store
            src += 1;
            dst += 1;
        , width);
        /* *INDENT-ON* */
        src = (const __m128i *)((const Uint8 *)src + srcskip);
        dst = (__m128i *)((Uint8 *)dst + dstskip);
    }
}
#endif

#if defined(HAVE_NEON_INTRINSICS)
static SDL_INLINE void BG_NEON(const TTF_Image *image, Uint8 *destination, Sint32 srcskip, Uint32 dstskip)
{
    const Uint8 *src    = image->buffer;
    Uint8       *dst    = destination;
    Uint32       width  = image->width / 16;
    Uint32       height = image->rows;

    uint8x16_t s, d, r;

    while (height--) {
        /* *INDENT-OFF* */
        DUFFS_LOOP4(
            s = vld1q_u8(src);  // load
            d = vld1q_u8(dst);  // load
            r = vorrq_u8(d, s); // or
            vst1q_u8(dst, r);   // store
            src += 16;
            dst += 16;
        , width);
        /* *INDENT-ON* */
        src = (const Uint8 *)((const Uint8 *)src + srcskip);
        dst += dstskip;
    }
}
#endif

/* Underline and Strikethrough style. Draw a line at the given row. */
static void Draw_Line(TTF_Font *font, const SDL_Surface *textbuf, int column, int row, int line_width, int line_thickness, Uint32 color, const render_mode_t render_mode)
{
    int tmp    = row + line_thickness - textbuf->h;
    int x_offset = column * textbuf->format->BytesPerPixel;
    Uint8 *dst = (Uint8 *)textbuf->pixels + row * textbuf->pitch + x_offset;
#if TTF_USE_HARFBUZZ
    hb_direction_t hb_direction = font->hb_direction;

    if (hb_direction == HB_DIRECTION_INVALID) {
        hb_direction = g_hb_direction;
    }

    /* No Underline/Strikethrough style if direction is vertical */
    if (hb_direction == HB_DIRECTION_TTB || hb_direction == HB_DIRECTION_BTT) {
        return;
    }
#else
    (void) font;
#endif

    /* Not needed because of "font->height = SDL_max(font->height, bottom_row);".
     * But if you patch to render textshaping and break line in middle of a cluster,
     * (which is a bad usage and a corner case), you need this to prevent out of bounds.
     * You can get an "ystart" for the "whole line", which is different (and smaller)
     * than the ones of the "splitted lines". */
    if (tmp > 0) {
        line_thickness -= tmp;
    }
    /* Previous case also happens with SDF (render_sdf) , because 'spread' property
     * requires to increase 'ystart'
     * Check for valid value anyway.  */
    if (line_thickness <= 0) {
        return;
    }

    /* Wrapped mode with an unbroken line: 'line_width' is greater that 'textbuf->w' */
    line_width = SDL_min(line_width, textbuf->w);

    if (render_mode == RENDER_BLENDED || render_mode == RENDER_LCD) {
        while (line_thickness--) {
            SDL_memset4(dst, color, line_width);
            dst += textbuf->pitch;
        }
    } else {
        while (line_thickness--) {
            SDL_memset(dst, color, line_width);
            dst += textbuf->pitch;
        }
    }
}

static void clip_glyph(int *_x, int *_y, TTF_Image *image, const SDL_Surface *textbuf, int is_lcd)
{
    int above_w;
    int above_h;
    int x = *_x;
    int y = *_y;

    int srcbpp = 1;
    if (image->is_color || is_lcd) {
        srcbpp = 4;
    }

    /* Don't go below x=0 */
    if (x < 0) {
        int tmp = -x;
        x = 0;
        image->width  -= tmp;
        image->buffer += srcbpp * tmp;
    }
    /* Don't go above textbuf->w */
    above_w = x + image->width - textbuf->w;
    if (above_w > 0) {
        image->width -= above_w;
    }
    /* Don't go below y=0 */
    if (y < 0) {
        int tmp = -y;
        y = 0;
        image->rows   -= tmp;
        image->buffer += tmp * image->pitch;
    }
    /* Don't go above textbuf->h */
    above_h = y + image->rows - textbuf->h;
    if (above_h > 0) {
        image->rows -= above_h;
    }
    /* Could be negative if (x > textbuf->w), or if (x + width < 0) */
    image->width = SDL_max(0, image->width);
    image->rows  = SDL_max(0, image->rows);

    /* After 'image->width' clipping:
     * Make sure 'rows' is also 0, so it doesn't break USE_DUFFS_LOOP */
    if (image->width == 0) {
        image->rows = 0;
    }

    *_x = x;
    *_y = y;
}

/* Glyph width is rounded, dst addresses are aligned, src addresses are not aligned */
static int Get_Alignment(void)
{
#if defined(HAVE_NEON_INTRINSICS)
    if (hasNEON()) {
        return 16;
    }
#endif

#if defined(HAVE_SSE2_INTRINSICS)
    if (hasSSE2()) {
        return 16;
    }
#endif

#if defined(HAVE_BLIT_GLYPH_64)
    return 8;
#elif defined(HAVE_BLIT_GLYPH_32)
    return 4;
#else
    return 1;
#endif
}

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
#define BUILD_RENDER_LINE(NAME, IS_BLENDED, IS_BLENDED_OPAQUE, IS_LCD, WB_WP_WC, WS, BLIT_GLYPH_BLENDED_OPAQUE_OPTIM, BLIT_GLYPH_BLENDED_OPTIM, BLIT_GLYPH_OPTIM) \
                                                                                                                        \
static SDL_INLINE                                                                                                       \
int Render_Line_##NAME(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg)                     \
{                                                                                                                       \
    const int alignment = Get_Alignment() - 1;                                                                         \
    const int bpp = ((IS_BLENDED || IS_LCD) ? 4 : 1);                                                                   \
    unsigned int i;                                                                                                     \
    Uint8 fg_alpha = (fg ? fg->a : 0);                                                                                  \
    for (i = 0; i < font->pos_len; i++) {                                                                               \
        FT_UInt idx = font->pos_buf[i].index;                                                                           \
        int x       = font->pos_buf[i].x;                                                                               \
        int y       = font->pos_buf[i].y;                                                                               \
        TTF_Image *image;                                                                                               \
                                                                                                                        \
        if (Find_GlyphByIndex(font, idx, WB_WP_WC, WS, x & 63, NULL, &image) == 0) {                                    \
            int above_w, above_h;                                                                                       \
            Uint32 dstskip;                                                                                             \
            Sint32 srcskip; /* Can be negative */                                                                       \
            Uint8 *dst;                                                                                                 \
            int remainder;                                                                                              \
            Uint8 *saved_buffer = image->buffer;                                                                        \
            int saved_width = image->width;                                                                             \
            image->buffer += alignment;                                                                                 \
            /* Position updated after glyph rendering */                                                                \
            x = xstart + FT_FLOOR(x) + image->left;                                                                     \
            y = ystart + FT_FLOOR(y) - image->top;                                                                      \
                                                                                                                        \
            /* Make sure glyph is inside textbuf */                                                                     \
            above_w = x + image->width - textbuf->w;                                                                    \
            above_h = y + image->rows  - textbuf->h;                                                                    \
                                                                                                                        \
            if (x >= 0 && y >= 0 && above_w <= 0 && above_h <= 0) {                                                     \
                /* Most often, glyph is inside textbuf */                                                               \
                /* Compute dst */                                                                                       \
                dst  = (Uint8 *)textbuf->pixels + y * textbuf->pitch + x * bpp;                                         \
                /* Align dst, get remainder, shift & align glyph width */                                               \
                remainder = ((uintptr_t)dst & alignment) / bpp;                                                         \
                dst  = (Uint8 *)((uintptr_t)dst & ~alignment);                                                          \
                image->buffer -= remainder;                                                                             \
                image->width   = (image->width + remainder + alignment) & ~alignment;                                   \
                /* Compute srcskip, dstskip */                                                                          \
                srcskip = image->pitch - image->width;                                                                  \
                dstskip = textbuf->pitch - image->width * bpp;                                                          \
                /* Render glyph at (x, y) with optimized copy functions */                                              \
                if (IS_LCD) {                                                                                           \
                    image->buffer = saved_buffer;                                                                       \
                    image->buffer += alignment;                                                                         \
                    image->width = saved_width;                                                                         \
                    dst = (Uint8 *)textbuf->pixels + y * textbuf->pitch + x * bpp;                                      \
                    /* Compute srcskip, dstskip */                                                                      \
                    srcskip = image->pitch - 4 * image->width;                                                          \
                    dstskip = textbuf->pitch - image->width * bpp;                                                      \
                    BG_Blended_LCD(image, (Uint32 *)dst, srcskip, dstskip, fg);                                         \
                } else if (!IS_BLENDED || image->is_color == 0) {                                                       \
                    if (IS_BLENDED_OPAQUE) {                                                                            \
                        BLIT_GLYPH_BLENDED_OPAQUE_OPTIM(image, (Uint32 *)dst, srcskip, dstskip);                        \
                    } else if (IS_BLENDED) {                                                                            \
                        BLIT_GLYPH_BLENDED_OPTIM(image, (Uint32 *)dst, srcskip, dstskip, fg_alpha);                     \
                    } else {                                                                                            \
                        BLIT_GLYPH_OPTIM(image, dst, srcskip, dstskip);                                                 \
                    }                                                                                                   \
                } else if (IS_BLENDED && image->is_color) {                                                             \
                    image->buffer = saved_buffer;                                                                       \
                    image->buffer += alignment;                                                                         \
                    image->width = saved_width;                                                                         \
                    dst = (Uint8 *)textbuf->pixels + y * textbuf->pitch + x * bpp;                                      \
                    /* Compute srcskip, dstskip */                                                                      \
                    srcskip = image->pitch - 4 * image->width;                                                          \
                    dstskip = textbuf->pitch - image->width * bpp;                                                      \
                    BG_Blended_Color(image, (Uint32 *)dst, srcskip, dstskip, fg_alpha);                                 \
                }                                                                                                       \
                /* restore modification */                                                                              \
                image->width = saved_width;                                                                             \
            } else {                                                                                                    \
                /* Modify a copy, and clip it */                                                                        \
                TTF_Image image_clipped = *image;                                                                       \
                /* Intersect image glyph at (x,y) with textbuf */                                                       \
                clip_glyph(&x, &y, &image_clipped, textbuf, IS_LCD);                                                    \
                /* Compute dst */                                                                                       \
                dst = (Uint8 *)textbuf->pixels + y * textbuf->pitch + x * bpp;                                          \
                /* Compute srcskip, dstskip */                                                                          \
                srcskip = image_clipped.pitch - image_clipped.width;                                                    \
                dstskip = textbuf->pitch - image_clipped.width * bpp;                                                   \
                /* Render glyph at (x, y) */                                                                            \
                if (IS_LCD) {                                                                                           \
                    srcskip -= 3 * image_clipped.width;                                                                 \
                    BG_Blended_LCD(&image_clipped, (Uint32 *)dst, srcskip, dstskip, fg);                                \
                } else if (!IS_BLENDED || image->is_color == 0) {                                                       \
                    if (IS_BLENDED_OPAQUE) {                                                                            \
                        BG_Blended_Opaque(&image_clipped, (Uint32 *)dst, srcskip, dstskip);                             \
                    } else if (IS_BLENDED) {                                                                            \
                        BG_Blended(&image_clipped, (Uint32 *)dst, srcskip, dstskip, fg_alpha);                          \
                    } else {                                                                                            \
                        BG(&image_clipped, dst, srcskip, dstskip);                                                      \
                    }                                                                                                   \
                } else if (IS_BLENDED && image->is_color) {                                                             \
                    srcskip -= 3 * image_clipped.width;                                                                 \
                    BG_Blended_Color(&image_clipped, (Uint32 *)dst, srcskip, dstskip, fg_alpha);                        \
                }                                                                                                       \
            }                                                                                                           \
            image->buffer = saved_buffer;                                                                               \
        } else {                                                                                                        \
            return -1;                                                                                                  \
        }                                                                                                               \
    }                                                                                                                   \
                                                                                                                        \
    return 0;                                                                                                           \
}                                                                                                                       \
                                                                                                                        \

#define BITMAP  CACHED_BITMAP, 0, 0, 0
#define PIXMAP  0, CACHED_PIXMAP, 0, 0
#define COLOR   0, 0, CACHED_COLOR, 0
#define LCD     0, 0, 0, CACHED_LCD

#define SUBPIX  CACHED_SUBPIX

/* BUILD_RENDER_LINE(NAME, IS_BLENDED, IS_BLENDED_OPAQUE, WANT_BITMAP_PIXMAP_COLOR_LCD, WANT_SUBPIXEL, BLIT_GLYPH_BLENDED_OPAQUE_OPTIM, BLIT_GLYPH_BLENDED_OPTIM, BLIT_GLYPH_OPTIM) */

#if defined(HAVE_SSE2_INTRINSICS)
BUILD_RENDER_LINE(SSE_Shaded            , 0, 0, 0, PIXMAP, 0     ,                       ,                , BG_SSE     )
BUILD_RENDER_LINE(SSE_Blended           , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended_SSE ,            )
BUILD_RENDER_LINE(SSE_Blended_Opaque    , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque_SSE ,                ,            )
BUILD_RENDER_LINE(SSE_Solid             , 0, 0, 0, BITMAP, 0     ,                       ,                , BG_SSE     )
BUILD_RENDER_LINE(SSE_Shaded_SP         , 0, 0, 0, PIXMAP, SUBPIX,                       ,                , BG_SSE     )
BUILD_RENDER_LINE(SSE_Blended_SP        , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended_SSE ,            )
BUILD_RENDER_LINE(SSE_Blended_Opaque_SP , 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque_SSE ,                ,            )
BUILD_RENDER_LINE(SSE_LCD               , 0, 0, 1,    LCD, 0,                            ,                ,            )
BUILD_RENDER_LINE(SSE_LCD_SP            , 0, 0, 1,    LCD, SUBPIX,                       ,                ,            )
#endif

#if defined(HAVE_NEON_INTRINSICS)
BUILD_RENDER_LINE(NEON_Shaded           , 0, 0, 0, PIXMAP, 0     ,                       ,                , BG_NEON    )
BUILD_RENDER_LINE(NEON_Blended          , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended_NEON,            )
BUILD_RENDER_LINE(NEON_Blended_Opaque   , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque_NEON,                ,            )
BUILD_RENDER_LINE(NEON_Solid            , 0, 0, 0, BITMAP, 0     ,                       ,                , BG_NEON    )
BUILD_RENDER_LINE(NEON_Shaded_SP        , 0, 0, 0, PIXMAP, SUBPIX,                       ,                , BG_NEON    )
BUILD_RENDER_LINE(NEON_Blended_SP       , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended_NEON,            )
BUILD_RENDER_LINE(NEON_Blended_Opaque_SP, 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque_NEON,                ,            )
BUILD_RENDER_LINE(NEON_LCD              , 0, 0, 1,    LCD, 0     ,                       ,                ,            )
BUILD_RENDER_LINE(NEON_LCD_SP           , 0, 0, 1,    LCD, SUBPIX,                       ,                ,            )
#endif

#if defined(HAVE_BLIT_GLYPH_64)
BUILD_RENDER_LINE(64_Shaded             , 0, 0, 0, PIXMAP, 0     ,                       ,                , BG_64      )
BUILD_RENDER_LINE(64_Blended            , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended_32  ,            )
BUILD_RENDER_LINE(64_Blended_Opaque     , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque_32  ,                ,            )
BUILD_RENDER_LINE(64_Solid              , 0, 0, 0, BITMAP, 0     ,                       ,                , BG_64      )
BUILD_RENDER_LINE(64_Shaded_SP          , 0, 0, 0, PIXMAP, SUBPIX,                       ,                , BG_64      )
BUILD_RENDER_LINE(64_Blended_SP         , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended_32  ,            )
BUILD_RENDER_LINE(64_Blended_Opaque_SP  , 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque_32  ,                ,            )
BUILD_RENDER_LINE(64_LCD                , 0, 0, 1,    LCD, 0     ,                       ,                ,            )
BUILD_RENDER_LINE(64_LCD_SP             , 0, 0, 1,    LCD, SUBPIX,                       ,                ,            )
#elif defined(HAVE_BLIT_GLYPH_32)
BUILD_RENDER_LINE(32_Shaded             , 0, 0, 0, PIXMAP, 0     ,                       ,                , BG_32      )
BUILD_RENDER_LINE(32_Blended            , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended_32  ,            )
BUILD_RENDER_LINE(32_Blended_Opaque     , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque_32  ,                ,            )
BUILD_RENDER_LINE(32_Solid              , 0, 0, 0, BITMAP, 0     ,                       ,                , BG_32      )
BUILD_RENDER_LINE(32_Shaded_SP          , 0, 0, 0, PIXMAP, SUBPIX,                       ,                , BG_32      )
BUILD_RENDER_LINE(32_Blended_SP         , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended_32  ,            )
BUILD_RENDER_LINE(32_Blended_Opaque_SP  , 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque_32  ,                ,            )
BUILD_RENDER_LINE(32_LCD                , 0, 0, 1,    LCD, 0     ,                       ,                ,            )
BUILD_RENDER_LINE(32_LCD_SP             , 0, 0, 1,    LCD, SUBPIX,                       ,                ,            )
#else
BUILD_RENDER_LINE(8_Shaded              , 0, 0, 0, PIXMAP, 0     ,                       ,                , BG         )
BUILD_RENDER_LINE(8_Blended             , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended     ,            )
BUILD_RENDER_LINE(8_Blended_Opaque      , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque     ,                ,            )
BUILD_RENDER_LINE(8_Solid               , 0, 0, 0, BITMAP, 0     ,                       ,                , BG         )
BUILD_RENDER_LINE(8_Shaded_SP           , 0, 0, 0, PIXMAP, SUBPIX,                       ,                , BG         )
BUILD_RENDER_LINE(8_Blended_SP          , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended     ,            )
BUILD_RENDER_LINE(8_Blended_Opaque_SP   , 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque     ,                ,            )
BUILD_RENDER_LINE(8_LCD                 , 0, 0, 1,    LCD, 0     ,                       ,                ,            )
BUILD_RENDER_LINE(8_LCD_SP              , 0, 0, 1,    LCD, SUBPIX,                       ,                ,            )
#endif


#if TTF_USE_SDF
static int (*Render_Line_SDF_Shaded)(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg) = NULL;
BUILD_RENDER_LINE(SDF_Blended           , 1, 0, 0,  COLOR, 0     ,                       , BG_Blended_SDF ,            )
BUILD_RENDER_LINE(SDF_Blended_Opaque    , 1, 1, 0,  COLOR, 0     , BG_Blended_Opaque_SDF ,                ,            )
static int (*Render_Line_SDF_Solid)(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg) = NULL;
static int (*Render_Line_SDF_Shaded_SP)(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg) = NULL;
BUILD_RENDER_LINE(SDF_Blended_SP        , 1, 0, 0,  COLOR, SUBPIX,                       , BG_Blended_SDF ,            )
BUILD_RENDER_LINE(SDF_Blended_Opaque_SP , 1, 1, 0,  COLOR, SUBPIX, BG_Blended_Opaque_SDF ,                ,            )
static int (*Render_Line_SDF_LCD)(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg) = NULL;
static int (*Render_Line_SDF_LCD_SP)(TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color *fg) = NULL;
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static SDL_INLINE int Render_Line(const render_mode_t render_mode, int subpixel, TTF_Font *font, SDL_Surface *textbuf, int xstart, int ystart, SDL_Color fg)
{
    /* Render line (pos_buf) to textbuf at (xstart, ystart) */

    /* Subpixel with RENDER_SOLID doesn't make sense. */
    /* (and 'cached->subpixel.translation' would need to distinguish bitmap/pixmap). */
    int is_opaque = (fg.a == SDL_ALPHA_OPAQUE);

#define Call_Specific_Render_Line(NAME)                                                                     \
        if (render_mode == RENDER_SHADED) {                                                                 \
            if (subpixel == 0) {                                                                            \
                return Render_Line_##NAME##_Shaded(font, textbuf, xstart, ystart, NULL);                    \
            } else {                                                                                        \
                return Render_Line_##NAME##_Shaded_SP(font, textbuf, xstart, ystart, NULL);                 \
            }                                                                                               \
        } else if (render_mode == RENDER_BLENDED) {                                                         \
            if (is_opaque) {                                                                                \
                if (subpixel == 0) {                                                                        \
                    return Render_Line_##NAME##_Blended_Opaque(font, textbuf, xstart, ystart, NULL);        \
                } else {                                                                                    \
                    return Render_Line_##NAME##_Blended_Opaque_SP(font, textbuf, xstart, ystart, NULL);     \
                }                                                                                           \
            } else {                                                                                        \
                if (subpixel == 0) {                                                                        \
                    return Render_Line_##NAME##_Blended(font, textbuf, xstart, ystart, &fg);                \
                } else {                                                                                    \
                    return Render_Line_##NAME##_Blended_SP(font, textbuf, xstart, ystart, &fg);             \
                }                                                                                           \
            }                                                                                               \
        } else if (render_mode == RENDER_LCD) {                                                             \
            if (subpixel == 0) {                                                                            \
                return Render_Line_##NAME##_LCD(font, textbuf, xstart, ystart, &fg);                        \
            } else {                                                                                        \
                return Render_Line_##NAME##_LCD_SP(font, textbuf, xstart, ystart, &fg);                     \
            }                                                                                               \
        } else {                                                                                            \
            return Render_Line_##NAME##_Solid(font, textbuf, xstart, ystart, NULL);                         \
        }

#if TTF_USE_SDF
    if (font->render_sdf && render_mode == RENDER_BLENDED) {
        Call_Specific_Render_Line(SDF)
    }
#endif

#if defined(HAVE_NEON_INTRINSICS)
    if (hasNEON()) {
        Call_Specific_Render_Line(NEON)
    }
#endif
#if defined(HAVE_SSE2_INTRINSICS)
    if (hasSSE2()) {
        Call_Specific_Render_Line(SSE)
    }
#endif
#if defined(HAVE_BLIT_GLYPH_64)
    Call_Specific_Render_Line(64)
#elif defined(HAVE_BLIT_GLYPH_32)
    Call_Specific_Render_Line(32)
#else
    Call_Specific_Render_Line(8)
#endif
}

#ifndef SIZE_MAX
# define SIZE_MAX ((size_t)(-1))
#endif

#if !SDL_VERSION_ATLEAST(2, 23, 1)
SDL_FORCE_INLINE int
compat_size_add_overflow (size_t a, size_t b, size_t *ret)
{
    if (b > SIZE_MAX - a) {
        return -1;
    }
    *ret = a + b;
    return 0;
}

SDL_FORCE_INLINE int
compat_size_mul_overflow (size_t a, size_t b, size_t *ret)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return -1;
    }
    *ret = a * b;
    return 0;
}

#define SDL_size_add_overflow(a, b, r) compat_size_add_overflow(a, b, r)
#define SDL_size_mul_overflow(a, b, r) compat_size_mul_overflow(a, b, r)
#endif /* SDL < 2.23.1 */

/* Create a surface with memory:
 * - pitch is rounded to alignment
 * - address is aligned
 *
 * If format is 4 bytes per pixel, bgcolor is used to initialize each
 * 4-byte word in the image data.
 *
 * Otherwise, the low byte of format is used to initialize each byte
 * in the image data.
 */
static SDL_Surface *AllocateAlignedPixels(size_t width, size_t height, SDL_PixelFormatEnum format, Uint32 bgcolor)
{
    const size_t alignment = Get_Alignment() - 1;
    const size_t bytes_per_pixel = SDL_BYTESPERPIXEL(format);
    SDL_Surface *textbuf = NULL;
    size_t size;
    size_t data_bytes;
    void *pixels, *ptr;
    size_t pitch;

    /* Worst case at the end of line pulling 'alignment' extra blank pixels */
    if (width > SDL_MAX_SINT32 ||
        height > SDL_MAX_SINT32 ||
        SDL_size_add_overflow(width, alignment, &pitch) ||
        SDL_size_mul_overflow(pitch, bytes_per_pixel, &pitch) ||
        SDL_size_add_overflow(pitch, alignment, &pitch) ||
        pitch > SDL_MAX_SINT32) {
        return NULL;
    }
    pitch &= ~alignment;

    if (SDL_size_mul_overflow(height, pitch, &data_bytes) ||
        SDL_size_add_overflow(data_bytes, sizeof (void *) + alignment, &size) ||
        size > SDL_MAX_SINT32) {
        /* Overflow... */
        return NULL;
    }

    ptr = SDL_malloc(size);
    if (ptr == NULL) {
        return NULL;
    }

    /* address is aligned */
    pixels = (void *)(((uintptr_t)ptr + sizeof(void *) + alignment) & ~alignment);
    ((void **)pixels)[-1] = ptr;

    textbuf = SDL_CreateRGBSurfaceWithFormatFrom(pixels, (int)width, (int)height, 0, (int)pitch, format);
    if (textbuf == NULL) {
        SDL_free(ptr);
        return NULL;
    }

    /* Let SDL handle the memory allocation */
    textbuf->flags &= ~SDL_PREALLOC;
    textbuf->flags |= SDL_SIMD_ALIGNED;

    if (bytes_per_pixel == 4) {
        SDL_memset4(pixels, bgcolor, data_bytes / 4);
    }
    else {
        SDL_memset(pixels, (bgcolor & 0xff), data_bytes);
    }

    return textbuf;
}

static SDL_Surface* Create_Surface_Solid(int width, int height, SDL_Color fg, Uint32 *color)
{
    SDL_Surface *textbuf = AllocateAlignedPixels(width, height, SDL_PIXELFORMAT_INDEX8, 0);
    if (textbuf == NULL) {
        return NULL;
    }

    /* Underline/Strikethrough color style */
    *color = 1;

    /* Fill the palette: 1 is foreground */
    {
        SDL_Palette *palette = textbuf->format->palette;
        palette->colors[0].r = 255 - fg.r;
        palette->colors[0].g = 255 - fg.g;
        palette->colors[0].b = 255 - fg.b;
        palette->colors[1].r = fg.r;
        palette->colors[1].g = fg.g;
        palette->colors[1].b = fg.b;
        palette->colors[1].a = fg.a;
    }

    SDL_SetColorKey(textbuf, SDL_TRUE, 0);

    return textbuf;
}

static SDL_Surface* Create_Surface_Shaded(int width, int height, SDL_Color fg, SDL_Color bg, Uint32 *color)
{
    SDL_Surface *textbuf = AllocateAlignedPixels(width, height, SDL_PIXELFORMAT_INDEX8, 0);
    Uint8 bg_alpha = bg.a;
    if (textbuf == NULL) {
        return NULL;
    }

    /* Underline/Strikethrough color style */
    *color = NUM_GRAYS - 1;

    /* Support alpha blending */
    if (fg.a != SDL_ALPHA_OPAQUE || bg.a != SDL_ALPHA_OPAQUE) {
        SDL_SetSurfaceBlendMode(textbuf, SDL_BLENDMODE_BLEND);

        /* Would disturb alpha palette */
        if (bg.a == SDL_ALPHA_OPAQUE) {
            bg.a = 0;
        }
    }

    /* Fill the palette with NUM_GRAYS levels of shading from bg to fg */
    {
        SDL_Palette *palette = textbuf->format->palette;
        int rdiff  = fg.r - bg.r;
        int gdiff  = fg.g - bg.g;
        int bdiff  = fg.b - bg.b;
        int adiff  = fg.a - bg.a;
        int sign_r = (rdiff >= 0) ? 1 : 255;
        int sign_g = (gdiff >= 0) ? 1 : 255;
        int sign_b = (bdiff >= 0) ? 1 : 255;
        int sign_a = (adiff >= 0) ? 1 : 255;
        int i;

        for (i = 0; i < NUM_GRAYS; ++i) {
            /* Compute color[i] = (i * color_diff / 255) */
            int tmp_r = i * rdiff;
            int tmp_g = i * gdiff;
            int tmp_b = i * bdiff;
            int tmp_a = i * adiff;
            palette->colors[i].r = (Uint8)(bg.r + DIVIDE_BY_255_SIGNED(tmp_r, sign_r));
            palette->colors[i].g = (Uint8)(bg.g + DIVIDE_BY_255_SIGNED(tmp_g, sign_g));
            palette->colors[i].b = (Uint8)(bg.b + DIVIDE_BY_255_SIGNED(tmp_b, sign_b));
            palette->colors[i].a = (Uint8)(bg.a + DIVIDE_BY_255_SIGNED(tmp_a, sign_a));
        }

        /* Make sure background has the correct alpha value */
        palette->colors[0].a = bg_alpha;
    }

    return textbuf;
}

static SDL_Surface *Create_Surface_Blended(int width, int height, SDL_Color fg, Uint32 *color)
{
    SDL_Surface *textbuf = NULL;
    Uint32 bgcolor;

    /* Background color: initialize with fg and 0 alpha */
    bgcolor = (fg.r << 16) | (fg.g << 8) | fg.b;

    /* Underline/Strikethrough color style */
    *color = bgcolor | ((Uint32)fg.a << 24);

    /* Create the target surface if required */
    if (width != 0) {
        textbuf = AllocateAlignedPixels(width, height, SDL_PIXELFORMAT_ARGB8888, bgcolor);
        if (textbuf == NULL) {
            return NULL;
        }

        /* Support alpha blending */
        if (fg.a != SDL_ALPHA_OPAQUE) {
            SDL_SetSurfaceBlendMode(textbuf, SDL_BLENDMODE_BLEND);
        }
    }

    return textbuf;
}

static SDL_Surface* Create_Surface_LCD(int width, int height, SDL_Color fg, SDL_Color bg, Uint32 *color)
{
    SDL_Surface *textbuf = NULL;
    Uint32 bgcolor;

    /* Background color */
    bgcolor = (((Uint32)bg.a) << 24) | (bg.r << 16) | (bg.g << 8) | bg.b;

    /* Underline/Strikethrough color style */
    *color = (((Uint32)bg.a) << 24) | (fg.r << 16) | (fg.g << 8) | fg.b;

    /* Create the target surface if required */
    if (width != 0) {
        textbuf = AllocateAlignedPixels(width, height, SDL_PIXELFORMAT_ARGB8888, bgcolor);
        if (textbuf == NULL) {
            return NULL;
        }

        /* Support alpha blending */
        if (bg.a != SDL_ALPHA_OPAQUE) {
            SDL_SetSurfaceBlendMode(textbuf, SDL_BLENDMODE_BLEND);
        }
    }

    return textbuf;
}


/* rcg06192001 get linked library's version. */
const SDL_version* TTF_Linked_Version(void)
{
    static SDL_version linked_version;
    SDL_TTF_VERSION(&linked_version);
    return &linked_version;
}

/* This function tells the library whether UNICODE text is generally
   byteswapped.  A UNICODE BOM character at the beginning of a string
   will override this setting for that string.  */
void TTF_ByteSwappedUNICODE(SDL_bool swapped)
{
    TTF_byteswapped = swapped;
}

#if defined(USE_FREETYPE_ERRORS)
static void TTF_SetFTError(const char *msg, FT_Error error)
{
#undef FTERRORS_H_
#define FT_ERRORDEF(e, v, s)    { e, s },
#define FT_ERROR_START_LIST     {
#define FT_ERROR_END_LIST       { 0, NULL } };
    const struct
    {
      int          err_code;
      const char  *err_msg;
    } ft_errors[] =
#include FT_ERRORS_H

    unsigned int i;
    const char *err_msg = NULL;

    for (i = 0; i < sizeof (ft_errors) / sizeof (ft_errors[0]); ++i) {
        if (error == ft_errors[i].err_code) {
            err_msg = ft_errors[i].err_msg;
            break;
        }
    }
    if (!err_msg) {
        err_msg = "unknown FreeType error";
    }
    TTF_SetError("%s: %s", msg, err_msg);
}
#else
#define TTF_SetFTError(msg, error)    TTF_SetError(msg)
#endif /* USE_FREETYPE_ERRORS */

int TTF_Init(void)
{
    int status = 0;

/* Some debug to know how it gets compiled */
#if 0
    int duffs = 0, sse2 = 0, neon = 0, compil_sse2 = 0, compil_neon = 0;
#  if defined(USE_DUFFS_LOOP)
    duffs = 1;
#  endif
#  if defined(HAVE_SSE2_INTRINSICS)
    sse2 = hasSSE2();
    compil_sse2 = 1;
#  endif
#  if defined(HAVE_NEON_INTRINSICS)
    neon = hasNEON();
    compil_neon = 1;
#  endif
    SDL_Log("SDL_ttf: hasSSE2=%d hasNEON=%d alignment=%d duffs_loop=%d compil_sse2=%d compil_neon=%d",
            sse2, neon, Get_Alignment(), duffs, compil_sse2, compil_neon);

    SDL_Log("Sizeof TTF_Image: %d c_glyph: %d TTF_Font: %d", sizeof (TTF_Image), sizeof (c_glyph), sizeof (TTF_Font));
#endif

    if (!TTF_initialized) {
        FT_Error error = FT_Init_FreeType(&library);
        if (error) {
            TTF_SetFTError("Couldn't init FreeType engine", error);
            status = -1;
        }
    }
    if (status == 0) {
        ++TTF_initialized;
#if TTF_USE_SDF
#  if 0
        /* Set various properties of the renderers. */
        int spread = 4;
        int overlaps = 0;
        FT_Property_Set( library, "bsdf", "spread", &spread);
        FT_Property_Set( library, "sdf", "spread", &spread);
        FT_Property_Set( library, "sdf", "overlaps", &overlaps);
#  endif
#endif
    }
    return status;
}

SDL_COMPILE_TIME_ASSERT(FT_Int, sizeof(int) == sizeof(FT_Int)); /* just in case. */
void TTF_GetFreeTypeVersion(int *major, int *minor, int *patch)
{
    FT_Library_Version(library, major, minor, patch);
}

void TTF_GetHarfBuzzVersion(int *major, int *minor, int *patch)
{
    unsigned int hb_major = 0;
    unsigned int hb_minor = 0;
    unsigned int hb_micro = 0;

#if TTF_USE_HARFBUZZ
    hb_version(&hb_major, &hb_minor, &hb_micro);
#endif
    if (major) {
        *major = (int)hb_major;
    }
    if (minor) {
        *minor = (int)hb_minor;
    }
    if (patch) {
        *patch = (int)hb_micro;
    }
}

static unsigned long RWread(
    FT_Stream stream,
    unsigned long offset,
    unsigned char *buffer,
    unsigned long count
)
{
    SDL_RWops *src;

    src = (SDL_RWops *)stream->descriptor.pointer;
    SDL_RWseek(src, (int)offset, RW_SEEK_SET);
    if (count == 0) {
        return 0;
    }
    return (unsigned long)SDL_RWread(src, buffer, 1, (int)count);
}

TTF_Font* TTF_OpenFontIndexDPIRW(SDL_RWops *src, int freesrc, int ptsize, long index, unsigned int hdpi, unsigned int vdpi)
{
    TTF_Font *font;
    FT_Error error;
    FT_Face face;
    FT_Stream stream;
    FT_CharMap found;
    Sint64 position;
    int i;

    if (!TTF_initialized) {
        TTF_SetError("Library not initialized");
        if (src && freesrc) {
            SDL_RWclose(src);
        }
        return NULL;
    }

    if (!src) {
        TTF_SetError("Passed a NULL font source");
        return NULL;
    }

    /* Check to make sure we can seek in this stream */
    position = SDL_RWtell(src);
    if (position < 0) {
        TTF_SetError("Can't seek in stream");
        if (freesrc) {
            SDL_RWclose(src);
        }
        return NULL;
    }

    font = (TTF_Font *)SDL_malloc(sizeof (*font));
    if (font == NULL) {
        TTF_SetError("Out of memory");
        if (freesrc) {
            SDL_RWclose(src);
        }
        return NULL;
    }
    SDL_memset(font, 0, sizeof (*font));

    font->src = src;
    font->freesrc = freesrc;

    stream = (FT_Stream)SDL_malloc(sizeof (*stream));
    if (stream == NULL) {
        TTF_SetError("Out of memory");
        TTF_CloseFont(font);
        return NULL;
    }
    SDL_memset(stream, 0, sizeof (*stream));

    stream->read = RWread;
    stream->descriptor.pointer = src;
    stream->pos = (unsigned long)position;
    stream->size = (unsigned long)(SDL_RWsize(src) - position);

    font->args.flags = FT_OPEN_STREAM;
    font->args.stream = stream;

    error = FT_Open_Face(library, &font->args, index, &font->face);
    if (error || font->face == NULL) {
        TTF_SetFTError("Couldn't load font file", error);
        TTF_CloseFont(font);
        return NULL;
    }
    face = font->face;

    /* Set charmap for loaded font */
    found = 0;
#if 0 /* Font debug code */
    for (i = 0; i < face->num_charmaps; i++) {
        FT_CharMap charmap = face->charmaps[i];
        SDL_Log("Found charmap: platform id %d, encoding id %d", charmap->platform_id, charmap->encoding_id);
    }
#endif
    if (!found) {
        for (i = 0; i < face->num_charmaps; i++) {
            FT_CharMap charmap = face->charmaps[i];
            if (charmap->platform_id == 3 && charmap->encoding_id == 10) { /* UCS-4 Unicode */
                found = charmap;
                break;
            }
        }
    }
    if (!found) {
        for (i = 0; i < face->num_charmaps; i++) {
            FT_CharMap charmap = face->charmaps[i];
            if ((charmap->platform_id == 3 && charmap->encoding_id == 1) /* Windows Unicode */
             || (charmap->platform_id == 3 && charmap->encoding_id == 0) /* Windows Symbol */
             || (charmap->platform_id == 2 && charmap->encoding_id == 1) /* ISO Unicode */
             || (charmap->platform_id == 0)) { /* Apple Unicode */
                found = charmap;
                break;
            }
        }
    }
    if (found) {
        /* If this fails, continue using the default charmap */
        FT_Set_Charmap(face, found);
    }

    /* Set the default font style */
    font->style = TTF_STYLE_NORMAL;
    font->outline_val = 0;
    font->ft_load_target = FT_LOAD_TARGET_NORMAL;
    TTF_SetFontKerning(font, 1);

    font->pos_len = 0;
    font->pos_max = 16;
    font->pos_buf = (PosBuf_t *)SDL_malloc(font->pos_max * sizeof (font->pos_buf[0]));
    if (! font->pos_buf) {
        TTF_SetError("Out of memory");
        TTF_CloseFont(font);
        return NULL;
    }

#if TTF_USE_HARFBUZZ
    font->hb_font = hb_ft_font_create(face, NULL);
    if (font->hb_font == NULL) {
        TTF_SetError("Cannot create harfbuzz font");
        TTF_CloseFont(font);
        return NULL;
    }

    /* Default load-flags of hb_ft_font_create is no-hinting.
     * So unless you call hb_ft_font_set_load_flags to match what flags you use for rendering,
     * you will get mismatching advances and raster. */
    hb_ft_font_set_load_flags(font->hb_font, FT_LOAD_DEFAULT | font->ft_load_target);

    /* By default the script / direction are inherited from global variables */
    font->hb_script = HB_SCRIPT_INVALID;
    font->hb_direction = HB_DIRECTION_INVALID;
#endif

    if (TTF_SetFontSizeDPI(font, ptsize, hdpi, vdpi) < 0) {
        TTF_SetFTError("Couldn't set font size", error);
        TTF_CloseFont(font);
        return NULL;
    }
    return font;
}

int TTF_SetFontSizeDPI(TTF_Font *font, int ptsize, unsigned int hdpi, unsigned int vdpi)
{
    FT_Face face = font->face;
    FT_Error error;

    /* Make sure that our font face is scalable (global metrics) */
    if (FT_IS_SCALABLE(face)) {
        /* Set the character size using the provided DPI.  If a zero DPI
         * is provided, then the other DPI setting will be used.  If both
         * are zero, then Freetype's default 72 DPI will be used.  */
        error = FT_Set_Char_Size(face, 0, ptsize * 64, hdpi, vdpi);
        if (error) {
            TTF_SetFTError("Couldn't set font size", error);
            return -1;
        }
    } else {
        /* Non-scalable font case.  ptsize determines which family
         * or series of fonts to grab from the non-scalable format.
         * It is not the point size of the font.  */
        if (face->num_fixed_sizes <= 0) {
            TTF_SetError("Couldn't select size : no num_fixed_sizes");
            return -1;
        }

        /* within [0; num_fixed_sizes - 1] */
        ptsize = SDL_max(ptsize, 0);
        ptsize = SDL_min(ptsize, face->num_fixed_sizes - 1);

        error = FT_Select_Size(face, ptsize);
        if (error) {
            TTF_SetFTError("Couldn't select size", error);
            return -1;
        }
    }

    if (TTF_initFontMetrics(font) < 0) {
        TTF_SetError("Cannot initialize metrics");
        return -1;
    }

    Flush_Cache(font);

#if TTF_USE_HARFBUZZ
    /* Call when size or variations settings on underlying FT_Face change. */
    hb_ft_font_changed(font->hb_font);
#endif

    return 0;
}

int TTF_SetFontSize(TTF_Font *font, int ptsize)
{
    return TTF_SetFontSizeDPI(font, ptsize, 0, 0);
}

/* Update font parameter depending on a style change */
static int TTF_initFontMetrics(TTF_Font *font)
{
    FT_Face face = font->face;
    int underline_offset;

    /* Make sure that our font face is scalable (global metrics) */
    if (FT_IS_SCALABLE(face)) {
        /* Get the scalable font metrics for this font */
        FT_Fixed scale       = face->size->metrics.y_scale;
        font->ascent         = FT_CEIL(FT_MulFix(face->ascender, scale));
        font->descent        = FT_CEIL(FT_MulFix(face->descender, scale));
        font->height         = FT_CEIL(FT_MulFix(face->ascender - face->descender, scale));
        font->lineskip       = FT_CEIL(FT_MulFix(face->height, scale));
        underline_offset     = FT_FLOOR(FT_MulFix(face->underline_position, scale));
        font->line_thickness = FT_FLOOR(FT_MulFix(face->underline_thickness, scale));
    } else {
        /* Get the font metrics for this font, for the selected size */
        font->ascent         = FT_CEIL(face->size->metrics.ascender);
        font->descent        = FT_CEIL(face->size->metrics.descender);
        font->height         = FT_CEIL(face->size->metrics.height);
        font->lineskip       = FT_CEIL(face->size->metrics.height);
        /* face->underline_position and face->underline_height are only
         * relevant for scalable formats (see freetype.h FT_FaceRec) */
        underline_offset     = font->descent / 2;
        font->line_thickness = 1;
    }

    if (font->line_thickness < 1) {
        font->line_thickness = 1;
    }

    font->underline_top_row     = font->ascent - underline_offset - 1;
    font->strikethrough_top_row = font->height / 2;

    /* Adjust OutlineStyle, only for scalable fonts */
    /* TTF_Size(): increase w and h by 2 * outline_val, translate positionning by 1 * outline_val */
    if (font->outline_val > 0) {
        int fo = font->outline_val;
        font->line_thickness        += 2 * fo;
        font->underline_top_row     -= fo;
        font->strikethrough_top_row -= fo;
    }

    /* Robustness: no negative values allowed */
    font->underline_top_row     = SDL_max(0, font->underline_top_row);
    font->strikethrough_top_row = SDL_max(0, font->strikethrough_top_row);

    /* Update height according to the needs of the underline style */
    if (TTF_HANDLE_STYLE_UNDERLINE(font)) {
        int bottom_row = font->underline_top_row + font->line_thickness;
        font->height = SDL_max(font->height, bottom_row);
    }
    /* Update height according to the needs of the strikethrough style */
    if (TTF_HANDLE_STYLE_STRIKETHROUGH(font)) {
        int bottom_row = font->strikethrough_top_row + font->line_thickness;
        font->height = SDL_max(font->height, bottom_row);
    }

#if defined(DEBUG_FONTS)
    SDL_Log("Font metrics:");
    SDL_Log("ascent = %d, descent = %d", font->ascent, font->descent);
    SDL_Log("height = %d, lineskip = %d", font->height, font->lineskip);
    SDL_Log("underline_offset = %d, line_thickness = %d", underline_offset, font->line_thickness);
    SDL_Log("underline_top_row = %d, strikethrough_top_row = %d", font->underline_top_row, font->strikethrough_top_row);
    SDL_Log("scalable=%d fixed_sizes=%d", FT_IS_SCALABLE(face), FT_HAS_FIXED_SIZES(face));
#endif

    font->glyph_overhang = face->size->metrics.y_ppem / 10;

    return 0;
}

TTF_Font* TTF_OpenFontDPIRW( SDL_RWops *src, int freesrc, int ptsize, unsigned int hdpi, unsigned int vdpi )
{
    return TTF_OpenFontIndexDPIRW(src, freesrc, ptsize, 0, hdpi, vdpi);
}

TTF_Font* TTF_OpenFontIndexRW( SDL_RWops *src, int freesrc, int ptsize, long index )
{
    return TTF_OpenFontIndexDPIRW(src, freesrc, ptsize, index, 0, 0);
}

TTF_Font* TTF_OpenFontIndexDPI( const char *file, int ptsize, long index, unsigned int hdpi, unsigned int vdpi )
{
    SDL_RWops *rw = SDL_RWFromFile(file, "rb");
    if ( rw == NULL ) {
        return NULL;
    }
    return TTF_OpenFontIndexDPIRW(rw, 1, ptsize, index, hdpi, vdpi);
}

TTF_Font* TTF_OpenFontRW(SDL_RWops *src, int freesrc, int ptsize)
{
    return TTF_OpenFontIndexRW(src, freesrc, ptsize, 0);
}

TTF_Font* TTF_OpenFontDPI(const char *file, int ptsize, unsigned int hdpi, unsigned int vdpi)
{
    return TTF_OpenFontIndexDPI(file, ptsize, 0, hdpi, vdpi);
}

TTF_Font* TTF_OpenFontIndex(const char *file, int ptsize, long index)
{
    return TTF_OpenFontIndexDPI(file, ptsize, index, 0, 0);
}

TTF_Font* TTF_OpenFont(const char *file, int ptsize)
{
    return TTF_OpenFontIndex(file, ptsize, 0);
}

static void Flush_Glyph_Image(TTF_Image *image) {
    if (image->buffer) {
        SDL_free(image->buffer);
        image->buffer = NULL;
    }
}

static void Flush_Glyph(c_glyph *glyph)
{
    glyph->stored = 0;
    glyph->index = 0;
    Flush_Glyph_Image(&glyph->pixmap);
    Flush_Glyph_Image(&glyph->bitmap);
}

static void Flush_Cache(TTF_Font *font)
{
    int i;
    int size = sizeof (font->cache) / sizeof (font->cache[0]);

    for (i = 0; i < size; ++i) {
        if (font->cache[i].stored) {
            Flush_Glyph(&font->cache[i]);
        }
    }
}

static FT_Error Load_Glyph(TTF_Font *font, c_glyph *cached, int want, int translation)
{
    const int alignment = Get_Alignment() - 1;
    FT_GlyphSlot slot;
    FT_Error error;

    int ft_load = FT_LOAD_DEFAULT | font->ft_load_target;

#if TTF_USE_COLOR
    if (want & CACHED_COLOR) {
        ft_load |= FT_LOAD_COLOR;
    }
#endif

    error = FT_Load_Glyph(font->face, cached->index, ft_load);
    if (error) {
        goto ft_failure;
    }

    /* Get our glyph shortcut */
    slot = font->face->glyph;

    if (want & CACHED_LCD) {
        if (slot->format == FT_GLYPH_FORMAT_BITMAP) {
            TTF_SetError("LCD mode not possible with bitmap font");
            return -1;
        }
    }

    /* Get the glyph metrics, always needed */
    if (cached->stored == 0) {
        cached->sz_left  = slot->bitmap_left;
        cached->sz_top   = slot->bitmap_top;
        cached->sz_rows  = slot->bitmap.rows;
        cached->sz_width = slot->bitmap.width;

        /* Current version of freetype is 2.9.1, but on older freetype (2.8.1) this can be 0.
         * Try to get them from 'FT_Glyph_Metrics' */
        if (cached->sz_left == 0 && cached->sz_top == 0 && cached->sz_rows == 0 && cached->sz_width == 0) {
            FT_Glyph_Metrics *metrics = &slot->metrics;
            if (metrics) {
                int minx = FT_FLOOR(metrics->horiBearingX);
                int maxx = FT_CEIL(metrics->horiBearingX + metrics->width);
                int maxy = FT_FLOOR(metrics->horiBearingY);
                int miny = maxy - FT_CEIL(metrics->height);

                cached->sz_left  = minx;
                cached->sz_top   = maxy;
                cached->sz_rows  = maxy - miny;
                cached->sz_width = maxx - minx;
            }
        }

        /* All FP 26.6 are 'long' but 'int' should be engouh */
        cached->advance  = (int)slot->metrics.horiAdvance; /* FP 26.6 */

        if (font->render_subpixel == 0) {
            /* FT KERNING_MODE_SMART */
            cached->kerning_smart.rsb_delta = (int)slot->rsb_delta; /* FP 26.6 */
            cached->kerning_smart.lsb_delta = (int)slot->lsb_delta; /* FP 26.6 */
        } else {
            /* FT LCD_MODE_LIGHT_SUBPIXEL */
            cached->subpixel.lsb_minus_rsb  = (int)(slot->lsb_delta - slot->rsb_delta); /* FP 26.6 */
            cached->subpixel.translation    = 0; /* FP 26.6 */
        }

#if defined(DEBUG_FONTS)
        SDL_Log("Index=%d sz_left=%d sz_top=%d sz_width=%d sz_rows=%d advance=%d is_outline=%d is_bitmap=%d",
                cached->index, cached->sz_left, cached->sz_top, cached->sz_width, cached->sz_rows, cached->advance,
                slot->format == FT_GLYPH_FORMAT_OUTLINE, slot->format == FT_GLYPH_FORMAT_BITMAP);
#endif

        /* Adjust for bold text */
        if (TTF_HANDLE_STYLE_BOLD(font)) {
            cached->sz_width += font->glyph_overhang;
            cached->advance  += F26Dot6(font->glyph_overhang);
        }

        /* Adjust for italic text */
        if (TTF_HANDLE_STYLE_ITALIC(font) && slot->format == FT_GLYPH_FORMAT_OUTLINE) {
            cached->sz_width += (GLYPH_ITALICS * font->height) >> 16;
        }

        /* Adjust for subpixel */
        if (font->render_subpixel) {
            cached->sz_width += 1;
        }

        /* Adjust for SDF */
        if (font->render_sdf) {
            /* Default 'spread' property */
            cached->sz_width += 2 * 8;
            cached->sz_rows  += 2 * 8;
        }

        cached->stored |= CACHED_METRICS;
    }

    if (((want & CACHED_BITMAP) && !(cached->stored & CACHED_BITMAP)) ||
        ((want & CACHED_PIXMAP) && !(cached->stored & CACHED_PIXMAP)) ||
        ((want & CACHED_COLOR) && !(cached->stored & CACHED_COLOR)) ||
        ((want & CACHED_LCD) && !(cached->stored & CACHED_LCD)) ||
         (want & CACHED_SUBPIX)
       ) {
        const int  mono  = (want & CACHED_BITMAP);
        TTF_Image *dst   = (mono ? &cached->bitmap : &cached->pixmap);
        FT_Glyph   glyph = NULL;
        FT_Bitmap *src;
        FT_Render_Mode ft_render_mode;

        if (mono) {
            ft_render_mode = FT_RENDER_MODE_MONO;
        } else {
            ft_render_mode = FT_RENDER_MODE_NORMAL;
#if TTF_USE_SDF
            if ((want & CACHED_COLOR) && font->render_sdf) {
                ft_render_mode = FT_RENDER_MODE_SDF;
            }
#endif
            if ((want & CACHED_LCD)) {
                ft_render_mode = FT_RENDER_MODE_LCD;
            }
        }

        /* Subpixel translation, flush previous datas */
        if (want & CACHED_SUBPIX) {
            Flush_Glyph_Image(&cached->pixmap);
            FT_Outline_Translate(&slot->outline, translation, 0 );
            cached->subpixel.translation = translation;
        }

        /* Handle the italic style, only for scalable fonts */
        if (TTF_HANDLE_STYLE_ITALIC(font) && slot->format == FT_GLYPH_FORMAT_OUTLINE) {
            FT_Matrix shear;
            shear.xx = 1 << 16;
            shear.xy = GLYPH_ITALICS;
            shear.yx = 0;
            shear.yy = 1 << 16;
            FT_Outline_Transform(&slot->outline, &shear);
        }

        /* Render as outline */
        if ((font->outline_val > 0 && slot->format == FT_GLYPH_FORMAT_OUTLINE)
            || slot->format == FT_GLYPH_FORMAT_BITMAP) {

            FT_BitmapGlyph bitmap_glyph;

            error = FT_Get_Glyph(slot, &glyph);
            if (error) {
                goto ft_failure;
            }

            if (font->outline_val > 0) {
                FT_Stroker stroker;
                error = FT_Stroker_New(library, &stroker);
                if (error) {
                    goto ft_failure;
                }
                FT_Stroker_Set(stroker, font->outline_val * 64, FT_STROKER_LINECAP_ROUND, FT_STROKER_LINEJOIN_ROUND, 0);
                FT_Glyph_Stroke(&glyph, stroker, 1 /* delete the original glyph */);
                FT_Stroker_Done(stroker);
            }

            /* Render the glyph */
            error = FT_Glyph_To_Bitmap(&glyph, ft_render_mode, 0, 1);
            if (error) {
                FT_Done_Glyph(glyph);
                goto ft_failure;
            }

            /* Access bitmap content by typecasting */
            bitmap_glyph = (FT_BitmapGlyph) glyph;
            src          = &bitmap_glyph->bitmap;

            /* Get new metrics, from bitmap */
            dst->left   = bitmap_glyph->left;
            dst->top    = bitmap_glyph->top;
        } else {
            /* Render the glyph */
            error = FT_Render_Glyph(slot, ft_render_mode);
            if (error) {
                goto ft_failure;
            }

            /* Access bitmap from slot */
            src         = &slot->bitmap;

            /* Get new metrics, from slot */
            dst->left   = slot->bitmap_left;
            dst->top    = slot->bitmap_top;
        }

        /* Common metrics */
        dst->width  = src->width;
        dst->rows   = src->rows;
        dst->buffer = NULL;

        /* FT can make small size glyph of 'width == 0', and 'rows != 0'.
         * Make sure 'rows' is also 0, so it doesn't break USE_DUFFS_LOOP */
        if (dst->width == 0) {
            dst->rows = 0;
        }

        /* Adjust for bold text */
        if (TTF_HANDLE_STYLE_BOLD(font)) {
            dst->width += font->glyph_overhang;
        }

        /* Compute pitch: glyph is padded right to be able to read an 'aligned' size expanding on the right */
        dst->pitch = dst->width + alignment;
#if TTF_USE_COLOR
        if (src->pixel_mode == FT_PIXEL_MODE_BGRA) {
            dst->pitch += 3 * dst->width;
        }
#endif
        if (src->pixel_mode == FT_PIXEL_MODE_LCD) {
            dst->pitch += 3 * dst->width;
        }

        if (dst->rows != 0) {
            unsigned int i;

            /* Glyph buffer is NOT aligned,
             * Extra width so it can read an 'aligned' size expanding on the left */
            dst->buffer = (unsigned char *)SDL_malloc(alignment + dst->pitch * dst->rows);

            if (!dst->buffer) {
                error = FT_Err_Out_Of_Memory;
                goto ft_failure;
            }

            /* Memset */
            SDL_memset(dst->buffer, 0, alignment + dst->pitch * dst->rows);

            /* Shift, so that the glyph is decoded centered */
            dst->buffer += alignment;

            /* FT_Render_Glyph() and .fon fonts always generate a two-color (black and white)
             * glyphslot surface, even when rendered in FT_RENDER_MODE_NORMAL. */
            /* FT_IS_SCALABLE() means that the face contains outline glyphs, but does not imply
             * that outline is rendered as 8-bit grayscale, because embedded bitmap/graymap is
             * preferred (see FT_LOAD_DEFAULT section of FreeType2 API Reference).
             * FT_Render_Glyph() canreturn two-color bitmap or 4/16/256 color graymap
             * according to the format of embedded bitmap/graymap. */
            for (i = 0; i < (unsigned int)src->rows; i++) {
                unsigned char *srcp = src->buffer + i * src->pitch;
                unsigned char *dstp = dst->buffer + i * dst->pitch;
                unsigned int k, quotient, remainder;

                /* Decode exactly the needed size from src->width */
                if (src->pixel_mode == FT_PIXEL_MODE_MONO) {
                    quotient  = src->width / 8;
                    remainder = src->width & 0x7;
                } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY2) {
                    quotient  = src->width / 4;
                    remainder = src->width & 0x3;
                } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY4) {
                    quotient  = src->width / 2;
                    remainder = src->width & 0x1;
#if TTF_USE_COLOR
                } else if (src->pixel_mode == FT_PIXEL_MODE_BGRA) {
                    quotient  = src->width;
                    remainder = 0;
#endif
                } else if (src->pixel_mode == FT_PIXEL_MODE_LCD) {
                    quotient  = src->width / 3;
                    remainder = 0;
                } else {
                    quotient  = src->width;
                    remainder = 0;
                }

/* FT_RENDER_MODE_MONO and src->pixel_mode MONO */
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable:4127)
#endif
#define MONO_MONO(K_MAX)                                                    \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        *dstp++ = (c & 0x80) >> 7;                          \
                        c <<= 1;                                            \
                    }                                                       \
                }

/* FT_RENDER_MODE_MONO and src->pixel_mode GRAY2 */
#define MONO_GRAY2(K_MAX)                                                   \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        *dstp++ = (((c&0xA0) >> 6) >= 0x2) ? 1 : 0;         \
                        c <<= 2;                                            \
                    }                                                       \
                }

/* FT_RENDER_MODE_MONO and src->pixel_mode GRAY4 */
#define MONO_GRAY4(K_MAX)                                                   \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        *dstp++ = (((c&0xF0) >> 4) >= 0x8) ? 1 : 0;         \
                        c <<= 4;                                            \
                    }                                                       \
                }

/* FT_RENDER_MODE_NORMAL and src->pixel_mode MONO */
#define NORMAL_MONO(K_MAX)                                                  \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        if ((c&0x80) >> 7) {                                \
                            *dstp++ = NUM_GRAYS - 1;                        \
                        } else {                                            \
                            *dstp++ = 0x00;                                 \
                        }                                                   \
                        c <<= 1;                                            \
                    }                                                       \
                }

/* FT_RENDER_MODE_NORMAL and src->pixel_mode GRAY2 */
#define NORMAL_GRAY2(K_MAX)                                                 \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        if ((c&0xA0) >> 6) {                                \
                            *dstp++ = NUM_GRAYS * ((c&0xA0) >> 6) / 3 - 1;  \
                        } else {                                            \
                            *dstp++ = 0x00;                                 \
                        }                                                   \
                        c <<= 2;                                            \
                    }                                                       \
                }

/* FT_RENDER_MODE_NORMAL and src->pixel_mode GRAY4 */
#define NORMAL_GRAY4(K_MAX)                                                 \
                if ((K_MAX)) {                                              \
                    unsigned char c = *srcp++;                              \
                    for (k = 0; k < (K_MAX); ++k) {                         \
                        if ((c&0xF0) >> 4) {                                \
                            *dstp++ = NUM_GRAYS * ((c&0xF0) >> 4) / 15 - 1; \
                        } else {                                            \
                            *dstp++ = 0x00;                                 \
                        }                                                   \
                        c <<= 4;                                            \
                    }                                                       \
                }

                if (mono) {
                    if (src->pixel_mode == FT_PIXEL_MODE_MONO) {
                        while (quotient--) {
                            MONO_MONO(8);
                        }
                        MONO_MONO(remainder);
                    } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY2) {
                        while (quotient--) {
                            MONO_GRAY2(4);
                        }
                        MONO_GRAY2(remainder);
                    } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY4) {
                        while (quotient--) {
                            MONO_GRAY4(2);
                        }
                        MONO_GRAY4(remainder);
                    } else {
                        while (quotient--) {
                            unsigned char c = *srcp++;
                            *dstp++ = (c >= 0x80) ? 1 : 0;
                        }
                    }
                } else if (src->pixel_mode == FT_PIXEL_MODE_MONO) {
                    /* This special case wouldn't be here if the FT_Render_Glyph()
                     * function wasn't buggy when it tried to render a .fon font with 256
                     * shades of gray.  Instead, it returns a black and white surface
                     * and we have to translate it back to a 256 gray shaded surface. */
                    while (quotient--) {
                        NORMAL_MONO(8);
                    }
                    NORMAL_MONO(remainder);
                } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY2) {
                    while (quotient--) {
                        NORMAL_GRAY2(4);
                    }
                    NORMAL_GRAY2(remainder);
                } else if (src->pixel_mode == FT_PIXEL_MODE_GRAY4) {
                    while (quotient--) {
                        NORMAL_GRAY4(2);
                    }
                    NORMAL_GRAY4(remainder);
#if TTF_USE_COLOR
                } else if (src->pixel_mode == FT_PIXEL_MODE_BGRA) {
                    SDL_memcpy(dstp, srcp, 4 * src->width);
#endif
                } else if (src->pixel_mode == FT_PIXEL_MODE_LCD) {
                    while (quotient--) {
                        Uint8 alpha = 0;
                        Uint8 r, g, b;
                        r = *srcp++;
                        g = *srcp++;
                        b = *srcp++;
                        *dstp++ = b;
                        *dstp++ = g;
                        *dstp++ = r;
                        *dstp++ = alpha;
                    }
                } else {
#if TTF_USE_SDF
                    if (ft_render_mode != FT_RENDER_MODE_SDF) {
                        SDL_memcpy(dstp, srcp, src->width);
                    } else {
                        int x;
                        for (x = 0; x < src->width; x++) {
                            Uint8 s = srcp[x];
                            Uint8 d;
                            if (s < 128) {
                                d = 256 - (128 - s) * 2;
                            } else {
                                d = 255;
                                /* some glitch ?
                                if (s == 255) {
                                    d = 0;
                                }*/
                            }
                            dstp[x] = d;
                        }
                    }
#else
                    SDL_memcpy(dstp, srcp, src->width);
#endif
                }
            }
        }
#ifdef _MSC_VER
#pragma warning(pop)
#endif

        /* Handle the bold style */
        if (TTF_HANDLE_STYLE_BOLD(font)) {
            int row;
            /* The pixmap is a little hard, we have to add and clamp */
            for (row = dst->rows - 1; row >= 0; --row) {
                Uint8 *pixmap = dst->buffer + row * dst->pitch;
                int col, offset;
                /* Minimal memset */
                /* SDL_memset(pixmap + dst->width - font->glyph_overhang, 0, font->glyph_overhang); */
                for (offset = 1; offset <= font->glyph_overhang; ++offset) {
                    for (col = dst->width - 1; col > 0; --col) {
                        if (mono) {
                            pixmap[col] |= pixmap[col-1];
                        } else {
                            int pixel = (pixmap[col] + pixmap[col-1]);
                            if (pixel > NUM_GRAYS - 1) {
                                pixel = NUM_GRAYS - 1;
                            }
                            pixmap[col] = (Uint8) pixel;
                        }
                    }
                }
            }
        }

        /* Shift back */
        if (dst->buffer) {
            dst->buffer -= alignment;
        }

#if TTF_USE_COLOR
        if (src->pixel_mode == FT_PIXEL_MODE_BGRA) {
            dst->is_color = 1;
        } else {
            dst->is_color = 0;
        }
#else
        dst->is_color = 0;
#endif

        /* Mark that we rendered this format */
        if (mono) {
            cached->stored |= CACHED_BITMAP;
        } else if (src->pixel_mode == FT_PIXEL_MODE_LCD) {
            cached->stored |= CACHED_LCD;
        } else {
#if TTF_USE_COLOR
            if (want & CACHED_COLOR) {
                cached->stored |= CACHED_COLOR;
                /* Most of the time, glyphs loaded with FT_LOAD_COLOR are non colored, so the cache is
                   also suitable for Shaded rendering (eg, loaded without FT_LOAD_COLOR) */
                if (dst->is_color == 0) {
                    cached->stored |= CACHED_PIXMAP;
                }
            } else {
                cached->stored |= CACHED_PIXMAP;
                /* If font has no color information, Shaded/Pixmap cache is also suitable for Blend/Color */
                if (!FT_HAS_COLOR(font->face)) {
                    cached->stored |= CACHED_COLOR;
                }
            }
#else
            cached->stored |= CACHED_COLOR | CACHED_PIXMAP;
#endif
        }

        /* Free outlined glyph */
        if (glyph) {
            FT_Done_Glyph(glyph);
        }
    }

    /* We're done, this glyph is cached since 'stored' is not 0 */
    return 0;

ft_failure:
    TTF_SetFTError("Couldn't find glyph", error);
    return -1;
}

static SDL_INLINE int Find_GlyphByIndex(TTF_Font *font, FT_UInt idx,
        int want_bitmap, int want_pixmap, int want_color, int want_lcd, int want_subpixel,
        int translation, c_glyph **out_glyph, TTF_Image **out_image)
{
    /* cache size is 256, get key by masking */
    c_glyph *glyph = &font->cache[idx & 0xff];

    if (out_glyph) {
        *out_glyph = glyph;
    }

    if (want_pixmap || want_color || want_lcd) {
        *out_image = &glyph->pixmap;
    }

    if (want_bitmap) {
        *out_image = &glyph->bitmap;
    }

    if (want_subpixel)
    {
        /* No a real cache, but if it always advances by integer pixels (eg translation 0 or same as previous),
         * this allows to render as fast as normal mode. */
        int retval;
        int want = CACHED_METRICS | want_bitmap | want_pixmap | want_color | want_lcd | want_subpixel;

        if (glyph->stored && glyph->index != idx) {
            Flush_Glyph(glyph);
        }

        if (glyph->subpixel.translation == translation) {
            want &= ~CACHED_SUBPIX;
        }

        if ((glyph->stored & want) == want) {
            return 0;
        }

        if (want_color || want_pixmap || want_lcd) {
            if (glyph->stored & (CACHED_COLOR|CACHED_PIXMAP|CACHED_LCD)) {
                Flush_Glyph(glyph);
            }
        }

        glyph->index = idx;
        retval = Load_Glyph(font, glyph, want, translation);
        if (retval == 0) {
            return 0;
        } else {
            return -1;
        }
    }
    else
    {
        int retval;
        const int want = CACHED_METRICS | want_bitmap | want_pixmap | want_color | want_lcd;

        /* Faster check as it gets inlined */
        if (want_pixmap) {
            if ((glyph->stored & CACHED_PIXMAP) && glyph->index == idx) {
                return 0;
            }
        } else if (want_bitmap) {
            if ((glyph->stored & CACHED_BITMAP) && glyph->index == idx) {
                return 0;
            }
        } else if (want_color) {
            if ((glyph->stored & CACHED_COLOR) && glyph->index == idx) {
                return 0;
            }
        } else if (want_lcd) {
            if ((glyph->stored & CACHED_LCD) && glyph->index == idx) {
                return 0;
            }
        } else {
            /* Get metrics */
            if (glyph->stored && glyph->index == idx) {
                return 0;
            }
        }

        /* Cache cannot contain both PIXMAP and COLOR (unless COLOR is actually not colored) and LCD
           So, if it's already used, clear it */
        if (want_color || want_pixmap || want_lcd) {
            if (glyph->stored & (CACHED_COLOR|CACHED_PIXMAP|CACHED_LCD)) {
                Flush_Glyph(glyph);
            }
        }

        if (glyph->stored && glyph->index != idx) {
            Flush_Glyph(glyph);
        }

        glyph->index = idx;
        retval = Load_Glyph(font, glyph, want, 0);
        if (retval == 0) {
            return 0;
        } else {
            return -1;
        }
    }
}

static SDL_INLINE FT_UInt get_char_index(TTF_Font *font, Uint32 ch)
{
    Uint32 cache_index_size = sizeof (font->cache_index) / sizeof (font->cache_index[0]);

    if (ch < cache_index_size) {
        FT_UInt idx = font->cache_index[ch];
        if (idx) {
            return idx;
        }
        idx = FT_Get_Char_Index(font->face, ch);
        font->cache_index[ch] = idx;
        return idx;
    }

    return FT_Get_Char_Index(font->face, ch);
}


static SDL_INLINE int Find_GlyphMetrics(TTF_Font *font, Uint32 ch, c_glyph **out_glyph)
{
    FT_UInt idx = get_char_index(font, ch);
    return Find_GlyphByIndex(font, idx, 0, 0, 0, 0, 0, 0, out_glyph, NULL);
}

void TTF_CloseFont(TTF_Font *font)
{
    if (font) {
#if TTF_USE_HARFBUZZ
        hb_font_destroy(font->hb_font);
#endif
        Flush_Cache(font);
        if (font->face) {
            FT_Done_Face(font->face);
        }
        if (font->args.stream) {
            SDL_free(font->args.stream);
        }
        if (font->freesrc) {
            SDL_RWclose(font->src);
        }
        if (font->pos_buf) {
            SDL_free(font->pos_buf);
        }
        SDL_free(font);
    }
}

/* Gets the number of bytes needed to convert a Latin-1 string to UTF-8 */
static size_t LATIN1_to_UTF8_len(const char *text)
{
    size_t bytes = 1;
    while (*text) {
        Uint8 ch = *(const Uint8 *)text++;
        if (ch <= 0x7F) {
            bytes += 1;
        } else {
            bytes += 2;
        }
    }
    return bytes;
}

/* Gets the number of bytes needed to convert a UCS2 string to UTF-8 */
static size_t UCS2_to_UTF8_len(const Uint16 *text)
{
    SDL_bool swapped = TTF_byteswapped;
    size_t bytes = 1;
    while (*text) {
        Uint16 ch = *text++;
        if (ch == UNICODE_BOM_NATIVE) {
            swapped = SDL_FALSE;
            continue;
        }
        if (ch == UNICODE_BOM_SWAPPED) {
            swapped = SDL_TRUE;
            continue;
        }
        if (swapped) {
            ch = SDL_Swap16(ch);
        }
        if (ch <= 0x7F) {
            bytes += 1;
        } else if (ch <= 0x7FF) {
            bytes += 2;
        } else {
            bytes += 3;
        }
    }
    return bytes;
}

/* Convert a Latin-1 string to a UTF-8 string */
static void LATIN1_to_UTF8(const char *src, Uint8 *dst)
{
    while (*src) {
        Uint8 ch = *(const Uint8 *)src++;
        if (ch <= 0x7F) {
            *dst++ = ch;
        } else {
            *dst++ = 0xC0 | ((ch >> 6) & 0x1F);
            *dst++ = 0x80 | (ch & 0x3F);
        }
    }
    *dst = '\0';
}

/* Convert a UCS-2 string to a UTF-8 string */
static void UCS2_to_UTF8(const Uint16 *src, Uint8 *dst)
{
    SDL_bool swapped = TTF_byteswapped;

    while (*src) {
        Uint16 ch = *src++;
        if (ch == UNICODE_BOM_NATIVE) {
            swapped = SDL_FALSE;
            continue;
        }
        if (ch == UNICODE_BOM_SWAPPED) {
            swapped = SDL_TRUE;
            continue;
        }
        if (swapped) {
            ch = SDL_Swap16(ch);
        }
        if (ch <= 0x7F) {
            *dst++ = (Uint8) ch;
        } else if (ch <= 0x7FF) {
            *dst++ = 0xC0 | (Uint8) ((ch >> 6) & 0x1F);
            *dst++ = 0x80 | (Uint8) (ch & 0x3F);
        } else {
            *dst++ = 0xE0 | (Uint8) ((ch >> 12) & 0x0F);
            *dst++ = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
            *dst++ = 0x80 | (Uint8) (ch & 0x3F);
        }
    }
    *dst = '\0';
}

/* Convert a unicode char to a UTF-8 string */
static SDL_bool Char_to_UTF8(Uint32 ch, Uint8 *dst)
{
    if (ch <= 0x7F) {
        *dst++ = (Uint8) ch;
    } else if (ch <= 0x7FF) {
        *dst++ = 0xC0 | (Uint8) ((ch >> 6) & 0x1F);
        *dst++ = 0x80 | (Uint8) (ch & 0x3F);
    } else if (ch <= 0xFFFF) {
        *dst++ = 0xE0 | (Uint8) ((ch >> 12) & 0x0F);
        *dst++ = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        *dst++ = 0x80 | (Uint8) (ch & 0x3F);
    } else if (ch <= 0x1FFFFF) {
        *dst++ = 0xF0 | (Uint8) ((ch >> 18) & 0x07);
        *dst++ = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        *dst++ = 0x80 | (Uint8) (ch & 0x3F);
    } else if (ch <= 0x3FFFFFF) {
        *dst++ = 0xF8 | (Uint8) ((ch >> 24) & 0x03);
        *dst++ = 0x80 | (Uint8) ((ch >> 18) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        *dst++ = 0x80 | (Uint8) (ch & 0x3F);
    } else if (ch < 0x7FFFFFFF) {
        *dst++ = 0xFC | (Uint8) ((ch >> 30) & 0x01);
        *dst++ = 0x80 | (Uint8) ((ch >> 24) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 18) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 12) & 0x3F);
        *dst++ = 0x80 | (Uint8) ((ch >> 6) & 0x3F);
        *dst++ = 0x80 | (Uint8) (ch & 0x3F);
    } else {
        TTF_SetError("Invalid character");
        return SDL_FALSE;
    }
    *dst = '\0';
    return SDL_TRUE;
}

/* Gets a unicode value from a UTF-8 encoded string
 * Ouputs increment to advance the string */
#define UNKNOWN_UNICODE 0xFFFD
static Uint32 UTF8_getch(const char *src, size_t srclen, int *inc)
{
    const Uint8 *p = (const Uint8 *)src;
    size_t left = 0;
    size_t save_srclen = srclen;
    SDL_bool overlong = SDL_FALSE;
    SDL_bool underflow = SDL_FALSE;
    Uint32 ch = UNKNOWN_UNICODE;

    if (srclen == 0) {
        return UNKNOWN_UNICODE;
    }
    if (p[0] >= 0xFC) {
        if ((p[0] & 0xFE) == 0xFC) {
            if (p[0] == 0xFC && (p[1] & 0xFC) == 0x80) {
                overlong = SDL_TRUE;
            }
            ch = (Uint32) (p[0] & 0x01);
            left = 5;
        }
    } else if (p[0] >= 0xF8) {
        if ((p[0] & 0xFC) == 0xF8) {
            if (p[0] == 0xF8 && (p[1] & 0xF8) == 0x80) {
                overlong = SDL_TRUE;
            }
            ch = (Uint32) (p[0] & 0x03);
            left = 4;
        }
    } else if (p[0] >= 0xF0) {
        if ((p[0] & 0xF8) == 0xF0) {
            if (p[0] == 0xF0 && (p[1] & 0xF0) == 0x80) {
                overlong = SDL_TRUE;
            }
            ch = (Uint32) (p[0] & 0x07);
            left = 3;
        }
    } else if (p[0] >= 0xE0) {
        if ((p[0] & 0xF0) == 0xE0) {
            if (p[0] == 0xE0 && (p[1] & 0xE0) == 0x80) {
                overlong = SDL_TRUE;
            }
            ch = (Uint32) (p[0] & 0x0F);
            left = 2;
        }
    } else if (p[0] >= 0xC0) {
        if ((p[0] & 0xE0) == 0xC0) {
            if ((p[0] & 0xDE) == 0xC0) {
                overlong = SDL_TRUE;
            }
            ch = (Uint32) (p[0] & 0x1F);
            left = 1;
        }
    } else {
        if ((p[0] & 0x80) == 0x00) {
            ch = (Uint32) p[0];
        }
    }
    --srclen;
    while (left > 0 && srclen > 0) {
        ++p;
        if ((p[0] & 0xC0) != 0x80) {
            ch = UNKNOWN_UNICODE;
            break;
        }
        ch <<= 6;
        ch |= (p[0] & 0x3F);
        --srclen;
        --left;
    }
    if (left > 0) {
        underflow = SDL_TRUE;
    }
    /* Technically overlong sequences are invalid and should not be interpreted.
       However, it doesn't cause a security risk here and I don't see any harm in
       displaying them. The application is responsible for any other side effects
       of allowing overlong sequences (e.g. string compares failing, etc.)
       See bug 1931 for sample input that triggers this.
    */
    /* if (overlong) return UNKNOWN_UNICODE; */

    (void) overlong;

    if (underflow ||
        (ch >= 0xD800 && ch <= 0xDFFF) ||
        (ch == 0xFFFE || ch == 0xFFFF) || ch > 0x10FFFF) {
        ch = UNKNOWN_UNICODE;
    }

    *inc = (int)(save_srclen - srclen);

    return ch;
}

int TTF_FontHeight(const TTF_Font *font)
{
    return font->height;
}

int TTF_FontAscent(const TTF_Font *font)
{
    return font->ascent + 2 * font->outline_val;
}

int TTF_FontDescent(const TTF_Font *font)
{
    return font->descent;
}

int TTF_FontLineSkip(const TTF_Font *font)
{
    return font->lineskip;
}

void TTF_SetFontLineSkip(TTF_Font *font, int lineskip)
{
    if (!font) {
        return;
    }

    font->lineskip = lineskip;
}

int TTF_GetFontKerning(const TTF_Font *font)
{
    return font->allow_kerning;
}

void TTF_SetFontKerning(TTF_Font *font, int allowed)
{
    font->allow_kerning = allowed;
#if TTF_USE_HARFBUZZ
    /* Harfbuzz can do kerning positioning even if the font hasn't the data */
#else
    font->use_kerning   = allowed && FT_HAS_KERNING(font->face);
#endif
}

long TTF_FontFaces(const TTF_Font *font)
{
    return font->face->num_faces;
}

int TTF_FontFaceIsFixedWidth(const TTF_Font *font)
{
    return FT_IS_FIXED_WIDTH(font->face);
}

const char *
TTF_FontFaceFamilyName(const TTF_Font *font)
{
    return font->face->family_name;
}

const char *
TTF_FontFaceStyleName(const TTF_Font *font)
{
    return font->face->style_name;
}

int TTF_GlyphIsProvided(TTF_Font *font, Uint16 ch)
{
    return (int)get_char_index(font, ch);
}

int TTF_GlyphIsProvided32(TTF_Font *font, Uint32 ch)
{
    return (int)get_char_index(font, ch);
}

int TTF_GlyphMetrics(TTF_Font *font, Uint16 ch,
                     int *minx, int *maxx, int *miny, int *maxy, int *advance)
{
    return TTF_GlyphMetrics32(font, ch, minx, maxx, miny, maxy, advance);
}

int TTF_GlyphMetrics32(TTF_Font *font, Uint32 ch,
                     int *minx, int *maxx, int *miny, int *maxy, int *advance)
{
    c_glyph *glyph;

    TTF_CHECK_POINTER(font, -1);

    if (Find_GlyphMetrics(font, ch, &glyph) < 0) {
        return -1;
    }

    if (minx) {
        *minx = glyph->sz_left;
    }
    if (maxx) {
        *maxx = glyph->sz_left + glyph->sz_width;
        *maxx += 2 * font->outline_val;
    }
    if (miny) {
        *miny = glyph->sz_top - glyph->sz_rows;
    }
    if (maxy) {
        *maxy = glyph->sz_top;
        *maxy += 2 * font->outline_val;
    }
    if (advance) {
        *advance = FT_CEIL(glyph->advance);
    }
    return 0;
}

int TTF_SetFontDirection(TTF_Font *font, TTF_Direction direction)
{
#if TTF_USE_HARFBUZZ
    hb_direction_t dir;
    if (direction == TTF_DIRECTION_LTR) {
        dir = HB_DIRECTION_LTR;
    } else if (direction == TTF_DIRECTION_RTL) {
        dir = HB_DIRECTION_RTL;
    } else if (direction == TTF_DIRECTION_TTB) {
        dir = HB_DIRECTION_BTT;
    } else if (direction == TTF_DIRECTION_BTT) {
        dir = HB_DIRECTION_TTB;
    } else {
        return -1;
    }
    font->hb_direction = dir;
    return 0;
#else
    (void) font;
    (void) direction;
    return -1;
#endif
}

int TTF_SetFontScriptName(TTF_Font *font, const char *script)
{
#if TTF_USE_HARFBUZZ
    Uint8 a, b, c, d;
    hb_script_t scr;

    if (script == NULL || SDL_strlen(script) != 4) {
        return -1;
    }

    a = script[0];
    b = script[1];
    c = script[2];
    d = script[3];

    scr = HB_TAG(a, b, c, d);
    font->hb_script = scr;
    return 0;
#else
    (void) font;
    (void) script;
    return -1;
#endif
}

static int TTF_Size_Internal(TTF_Font *font,
        const char *text, const str_type_t str_type,
        int *w, int *h, int *xstart, int *ystart,
        int measure_width, int *extent, int *count)
{
    int x = 0;
    int pos_x, pos_y;
    int minx = 0, maxx = 0;
    int miny = 0, maxy = 0;
    Uint8 *utf8_alloc = NULL;
    c_glyph *glyph;
#if TTF_USE_HARFBUZZ
    hb_direction_t hb_direction;
    hb_script_t hb_script;
    hb_buffer_t *hb_buffer = NULL;
    unsigned int g;
    unsigned int glyph_count;
    hb_glyph_info_t *hb_glyph_info;
    hb_glyph_position_t *hb_glyph_position;
    int y = 0;
    int advance_if_bold = 0;
#else
    size_t textlen;
    int skip_first = 1;
    FT_UInt prev_index = 0;
    FT_Pos  prev_delta = 0;
#endif
    int prev_advance = 0;

    /* Measurement mode */
    int char_count = 0;
    int current_width = 0;

    TTF_CHECK_INITIALIZED(-1);
    TTF_CHECK_POINTER(font, -1);
    TTF_CHECK_POINTER(text, -1);

    /* Convert input string to default encoding UTF-8 */
    if (str_type == STR_TEXT) {
        utf8_alloc = SDL_stack_alloc(Uint8, LATIN1_to_UTF8_len(text));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        LATIN1_to_UTF8(text, utf8_alloc);
        text = (const char *)utf8_alloc;
    } else if (str_type == STR_UNICODE) {
        const Uint16 *text16 = (const Uint16 *) text;
        utf8_alloc = SDL_stack_alloc(Uint8, UCS2_to_UTF8_len(text16));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        UCS2_to_UTF8(text16, utf8_alloc);
        text = (const char *)utf8_alloc;
    }

    maxy = font->height;

    /* Reset buffer */
    font->pos_len = 0;

#if TTF_USE_HARFBUZZ

    /* Adjust for bold text */
    if (TTF_HANDLE_STYLE_BOLD(font)) {
        advance_if_bold = F26Dot6(font->glyph_overhang);
    }

    /* Create a buffer for harfbuzz to use */
    hb_buffer = hb_buffer_create();
    if (hb_buffer == NULL) {
       TTF_SetError("Cannot create harfbuzz buffer");
       goto failure;
    }


    hb_direction = font->hb_direction;
    hb_script = font->hb_script;

    if (hb_script == HB_SCRIPT_INVALID) {
        hb_script = g_hb_script;
    }

    if (hb_direction == HB_DIRECTION_INVALID) {
        hb_direction = g_hb_direction;
    }

    /* Set global configuration */
    hb_buffer_set_direction(hb_buffer, hb_direction);
    hb_buffer_set_script(hb_buffer, hb_script);
    hb_buffer_guess_segment_properties(hb_buffer);

    /* Layout the text */
    hb_buffer_add_utf8(hb_buffer, text, -1, 0, -1);

    hb_feature_t userfeatures[1];
    userfeatures[0].tag = HB_TAG('k','e','r','n');
    userfeatures[0].value = font->allow_kerning;
    userfeatures[0].start = HB_FEATURE_GLOBAL_START;
    userfeatures[0].end = HB_FEATURE_GLOBAL_END;

    hb_shape(font->hb_font, hb_buffer, userfeatures, 1);

    /* Get the result */
    hb_glyph_info = hb_buffer_get_glyph_infos(hb_buffer, &glyph_count);
    hb_glyph_position = hb_buffer_get_glyph_positions(hb_buffer, &glyph_count);

    /* Load and render each character */
    for (g = 0; g < glyph_count; g++)
    {
        FT_UInt idx   = hb_glyph_info[g].codepoint;
        int x_advance = hb_glyph_position[g].x_advance;
        int y_advance = hb_glyph_position[g].y_advance;
        int x_offset  = hb_glyph_position[g].x_offset;
        int y_offset  = hb_glyph_position[g].y_offset;
#else
    /* Load each character and sum it's bounding box */
    textlen = SDL_strlen(text);
    while (textlen > 0) {
        int inc = 0;
        Uint32 c = UTF8_getch(text, textlen, &inc);
        FT_UInt idx = get_char_index(font, c);
        text += inc;
        textlen -= inc;

        if (c == UNICODE_BOM_NATIVE || c == UNICODE_BOM_SWAPPED) {
            continue;
        }
#endif
        if (Find_GlyphByIndex(font, idx, 0, 0, 0, 0, 0, 0, &glyph, NULL) < 0) {
            goto failure;
        }

        /* Realloc, if needed */
        if (font->pos_len >= font->pos_max) {
            PosBuf_t *saved = font->pos_buf;
            font->pos_max *= 2;
            font->pos_buf = (PosBuf_t *)SDL_realloc(font->pos_buf, font->pos_max * sizeof (font->pos_buf[0]));
            if (font->pos_buf == NULL) {
                font->pos_max /= 2;
                font->pos_buf = saved;
                TTF_SetError("Out of memory");
                goto failure;
            }
        }

#if TTF_USE_HARFBUZZ
        /* Compute positions */
        pos_x  = x                     + x_offset;
        pos_y  = y + F26Dot6(font->ascent) - y_offset;
        x     += x_advance + advance_if_bold;
        y     += y_advance;
#else
        /* Compute positions */
        x += prev_advance;
        prev_advance = glyph->advance;
        if (font->use_kerning) {
            if (prev_index && glyph->index) {
                FT_Vector delta;
                FT_Get_Kerning(font->face, prev_index, glyph->index, FT_KERNING_UNFITTED, &delta);
                x += delta.x;
            }
            prev_index = glyph->index;
        }
        /* FT SUBPIXEL : LCD_MODE_LIGHT_SUBPIXEL  */
        if (font->render_subpixel) {
            x += prev_delta;
            /* Increment by prev_glyph->lsb_delta - prev_glyph->rsb_delta; */
            prev_delta = glyph->subpixel.lsb_minus_rsb;
        } else {
            /* FT KERNING_MODE_SMART: Use `lsb_delta' and `rsb_delta' to improve integer positioning of glyphs */
            if (skip_first) {
                skip_first = 0;
            } else {
                if (prev_delta - glyph->kerning_smart.lsb_delta >  32 ) {
                    x -= 64;
                } else if (prev_delta - glyph->kerning_smart.lsb_delta < -31 ) {
                    x += 64;
                }
            }
            prev_delta = glyph->kerning_smart.rsb_delta;
            x = ((x + 32) & -64); /* ROUND() */
        }

        /* Compute positions where to copy the glyph bitmap */
        pos_x = x;
        pos_y = F26Dot6(font->ascent);
#endif
        /* Store things for Render_Line() */
        font->pos_buf[font->pos_len].x     = pos_x;
        font->pos_buf[font->pos_len].y     = pos_y;
        font->pos_buf[font->pos_len].index = idx;
        font->pos_len += 1;

        /* Compute previsionnal global bounding box */
        pos_x = FT_FLOOR(pos_x) + glyph->sz_left;
        pos_y = FT_FLOOR(pos_y) - glyph->sz_top;

        minx = SDL_min(minx, pos_x);
        maxx = SDL_max(maxx, pos_x + glyph->sz_width);
        miny = SDL_min(miny, pos_y);
        maxy = SDL_max(maxy, pos_y + glyph->sz_rows);

        /* Measurement mode */
        if (measure_width) {
            int cw = SDL_max(maxx, FT_FLOOR(x + prev_advance)) - minx;
            cw += 2 * font->outline_val;
            if (cw <= measure_width) {
                current_width = cw;
                char_count += 1;
            }
            if (cw >= measure_width) {
                break;
            }
        }
    }

    /* Allows to render a string with only one space (bug 4344). */
    maxx = SDL_max(maxx, FT_FLOOR(x + prev_advance));

    /* Initial x start position: often 0, except when a glyph would be written at
     * a negative position. In this case an offset is needed for the whole line. */
    if (xstart) {
        *xstart = (minx < 0)? -minx : 0;
        *xstart += font->outline_val;
        if (font->render_sdf) {
            *xstart += 8; /* Default 'spread' property */
        }
    }

    /* Initial y start: compensation for a negative y offset */
    if (ystart) {
        *ystart = (miny < 0)? -miny : 0;
        *ystart += font->outline_val;
        if (font->render_sdf) {
            *ystart += 8; /* Default 'spread' property */
        }
    }

    /* Fill the bounds rectangle */
    if (w) {
        *w = (maxx - minx);
        if (*w != 0) {
            *w += 2 * font->outline_val;
        }
    }
    if (h) {
        *h = (maxy - miny);
        *h += 2 * font->outline_val;
    }

    /* Measurement mode */
    if (measure_width) {
        if (extent) {
            *extent = current_width;
        }
        if (count) {
#if TTF_USE_HARFBUZZ
            if (char_count == glyph_count) {
                /* The higher level code doesn't know about ligatures,
                 * so if we've covered all the glyphs, report the full
                 * string length.
                 *
                 * If we have to line wrap somewhere in the middle, we
                 * might be off by the number of ligatures, but there
                 * isn't an easy way around that without using hb_buffer
                 * at that level instead.
                 */
                *count = (int)SDL_utf8strlen(text);
            } else
#endif
                *count = char_count;
        }
    }

#if TTF_USE_HARFBUZZ
    if (hb_buffer) {
        hb_buffer_destroy(hb_buffer);
    }
#endif
    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return 0;
failure:
#if TTF_USE_HARFBUZZ
    if (hb_buffer) {
        hb_buffer_destroy(hb_buffer);
    }
#endif
    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return -1;
}

int TTF_SizeText(TTF_Font *font, const char *text, int *w, int *h)
{
    return TTF_Size_Internal(font, text, STR_TEXT, w, h, NULL, NULL, NO_MEASUREMENT);
}

int TTF_SizeUTF8(TTF_Font *font, const char *text, int *w, int *h)
{
    return TTF_Size_Internal(font, text, STR_UTF8, w, h, NULL, NULL, NO_MEASUREMENT);
}

int TTF_SizeUNICODE(TTF_Font *font, const Uint16 *text, int *w, int *h)
{
    return TTF_Size_Internal(font, (const char *)text, STR_UNICODE, w, h, NULL, NULL, NO_MEASUREMENT);
}

int TTF_MeasureText(TTF_Font *font, const char *text, int width, int *extent, int *count)
{
    return TTF_Size_Internal(font, text, STR_TEXT, NULL, NULL, NULL, NULL, width, extent, count);
}

int TTF_MeasureUTF8(TTF_Font *font, const char *text, int width, int *extent, int *count)
{
    return TTF_Size_Internal(font, text, STR_UTF8, NULL, NULL, NULL, NULL, width, extent, count);
}

int TTF_MeasureUNICODE(TTF_Font *font, const Uint16 *text, int width, int *extent, int *count)
{
    return TTF_Size_Internal(font, (const char *)text, STR_UNICODE, NULL, NULL, NULL, NULL, width, extent, count);
}

static SDL_Surface* TTF_Render_Internal(TTF_Font *font, const char *text, const str_type_t str_type,
        SDL_Color fg, SDL_Color bg, const render_mode_t render_mode)
{
    Uint32 color;
    int xstart, ystart, width, height;
    SDL_Surface *textbuf = NULL;
    Uint8 *utf8_alloc = NULL;

    TTF_CHECK_INITIALIZED(NULL);
    TTF_CHECK_POINTER(font, NULL);
    TTF_CHECK_POINTER(text, NULL);

    if (render_mode == RENDER_LCD && !FT_IS_SCALABLE(font->face)) {
        TTF_SetError("LCD rendering is not available for non-scalable font");
        goto failure;
    }

    /* Convert input string to default encoding UTF-8 */
    if (str_type == STR_TEXT) {
        utf8_alloc = SDL_stack_alloc(Uint8, LATIN1_to_UTF8_len(text));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        LATIN1_to_UTF8(text, utf8_alloc);
        text = (const char *)utf8_alloc;
    } else if (str_type == STR_UNICODE) {
        const Uint16 *text16 = (const Uint16 *) text;
        utf8_alloc = SDL_stack_alloc(Uint8, UCS2_to_UTF8_len(text16));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        UCS2_to_UTF8(text16, utf8_alloc);
        text = (const char *)utf8_alloc;
    }

#if TTF_USE_SDF
    /* Invalid cache if we were using SDF */
    if (render_mode != RENDER_BLENDED) {
        if (font->render_sdf) {
            font->render_sdf = 0;
            Flush_Cache(font);
        }
    }
#endif

    /* Get the dimensions of the text surface */
    if ((TTF_Size_Internal(font, text, STR_UTF8, &width, &height, &xstart, &ystart, NO_MEASUREMENT) < 0) || !width) {
        TTF_SetError("Text has zero width");
        goto failure;
    }

    /* Support alpha blending */
    fg.a = fg.a ? fg.a : SDL_ALPHA_OPAQUE;
    bg.a = bg.a ? bg.a : SDL_ALPHA_OPAQUE;

    /* Create surface for rendering */
    if (render_mode == RENDER_SOLID) {
        textbuf = Create_Surface_Solid(width, height, fg, &color);
    } else if (render_mode == RENDER_SHADED) {
        textbuf = Create_Surface_Shaded(width, height, fg, bg, &color);
    } else if (render_mode == RENDER_BLENDED) {
        textbuf = Create_Surface_Blended(width, height, fg, &color);
    } else { /* render_mode == RENDER_LCD */
        textbuf = Create_Surface_LCD(width, height, fg, bg, &color);
    }

    if (textbuf == NULL) {
        goto failure;
    }

    /* Render one text line to textbuf at (xstart, ystart) */
    if (Render_Line(render_mode, font->render_subpixel, font, textbuf, xstart, ystart, fg) < 0) {
        goto failure;
    }

    /* Apply underline or strikethrough style, if needed */
    if (TTF_HANDLE_STYLE_UNDERLINE(font)) {
        Draw_Line(font, textbuf, 0, ystart + font->underline_top_row, width, font->line_thickness, color, render_mode);
    }

    if (TTF_HANDLE_STYLE_STRIKETHROUGH(font)) {
        Draw_Line(font, textbuf, 0, ystart + font->strikethrough_top_row, width, font->line_thickness, color, render_mode);
    }

    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return textbuf;
failure:
    if (textbuf) {
        SDL_FreeSurface(textbuf);
    }
    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return NULL;
}

SDL_Surface* TTF_RenderText_Solid(TTF_Font *font, const char *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, text, STR_TEXT, fg, fg /* unused */, RENDER_SOLID);
}

SDL_Surface* TTF_RenderUTF8_Solid(TTF_Font *font, const char *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, text, STR_UTF8, fg, fg /* unused */, RENDER_SOLID);
}

SDL_Surface* TTF_RenderUNICODE_Solid(TTF_Font *font, const Uint16 *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, (const char *)text, STR_UNICODE, fg, fg /* unused */, RENDER_SOLID);
}

SDL_Surface* TTF_RenderGlyph_Solid(TTF_Font *font, Uint16 ch, SDL_Color fg)
{
    return TTF_RenderGlyph32_Solid(font, ch, fg);
}

SDL_Surface* TTF_RenderGlyph32_Solid(TTF_Font *font, Uint32 ch, SDL_Color fg)
{
    Uint8 utf8[7];

    TTF_CHECK_POINTER(font, NULL);

    if (!Char_to_UTF8(ch, utf8)) {
        return NULL;
    }

    return TTF_RenderUTF8_Solid(font, (char *)utf8, fg);
}

SDL_Surface* TTF_RenderText_Shaded(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, text, STR_TEXT, fg, bg, RENDER_SHADED);
}

SDL_Surface* TTF_RenderUTF8_Shaded(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, text, STR_UTF8, fg, bg, RENDER_SHADED);
}

SDL_Surface* TTF_RenderUNICODE_Shaded(TTF_Font *font, const Uint16 *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, (const char *)text, STR_UNICODE, fg, bg, RENDER_SHADED);
}

SDL_Surface* TTF_RenderGlyph_Shaded(TTF_Font *font, Uint16 ch, SDL_Color fg, SDL_Color bg)
{
    return TTF_RenderGlyph32_Shaded(font, ch, fg, bg);
}

SDL_Surface* TTF_RenderGlyph32_Shaded(TTF_Font *font, Uint32 ch, SDL_Color fg, SDL_Color bg)
{
    Uint8 utf8[7];

    TTF_CHECK_POINTER(font, NULL);

    if (!Char_to_UTF8(ch, utf8)) {
        return NULL;
    }

    return TTF_RenderUTF8_Shaded(font, (char *)utf8, fg, bg);
}

SDL_Surface* TTF_RenderText_Blended(TTF_Font *font, const char *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, text, STR_TEXT, fg, fg /* unused */, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderUTF8_Blended(TTF_Font *font, const char *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, text, STR_UTF8, fg, fg /* unused */, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderUNICODE_Blended(TTF_Font *font, const Uint16 *text, SDL_Color fg)
{
    return TTF_Render_Internal(font, (const char *)text, STR_UNICODE, fg, fg /* unused */, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderText_LCD(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, text, STR_TEXT, fg, bg, RENDER_LCD);
}

SDL_Surface* TTF_RenderUTF8_LCD(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, text, STR_UTF8, fg, bg, RENDER_LCD);
}

SDL_Surface* TTF_RenderUNICODE_LCD(TTF_Font *font, const Uint16 *text, SDL_Color fg, SDL_Color bg)
{
    return TTF_Render_Internal(font, (const char *)text, STR_UNICODE, fg, bg, RENDER_LCD);
}

SDL_Surface* TTF_RenderGlyph_LCD(TTF_Font *font, Uint16 ch, SDL_Color fg, SDL_Color bg)
{
    return TTF_RenderGlyph32_LCD(font, ch, fg, bg);
}

SDL_Surface* TTF_RenderGlyph32_LCD(TTF_Font *font, Uint32 ch, SDL_Color fg, SDL_Color bg)
{
    Uint8 utf8[7];

    TTF_CHECK_POINTER(font, NULL);

    if (!Char_to_UTF8(ch, utf8)) {
        return NULL;
    }

    return TTF_RenderUTF8_LCD(font, (char *)utf8, fg, bg);
}

static SDL_bool CharacterIsDelimiter(Uint32 c)
{
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static SDL_bool CharacterIsNewLine(Uint32 c)
{
    if (c == '\n') {
        return SDL_TRUE;
    }
    return SDL_FALSE;
}

static SDL_Surface* TTF_Render_Wrapped_Internal(TTF_Font *font, const char *text, const str_type_t str_type,
        SDL_Color fg, SDL_Color bg, Uint32 wrapLength, const render_mode_t render_mode)
{
    Uint32 color;
    int width, height;
    SDL_Surface *textbuf = NULL;
    Uint8 *utf8_alloc = NULL;

    int i, numLines, rowHeight, lineskip;
    char **strLines = NULL, *text_cpy;

    TTF_CHECK_INITIALIZED(NULL);
    TTF_CHECK_POINTER(font, NULL);
    TTF_CHECK_POINTER(text, NULL);

    if (render_mode == RENDER_LCD && !FT_IS_SCALABLE(font->face)) {
        TTF_SetError("LCD rendering is not available for non-scalable font");
        goto failure;
    }

    /* Convert input string to default encoding UTF-8 */
    if (str_type == STR_TEXT) {
        utf8_alloc = SDL_stack_alloc(Uint8, LATIN1_to_UTF8_len(text));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        LATIN1_to_UTF8(text, utf8_alloc);
        text_cpy = (char *)utf8_alloc;
    } else if (str_type == STR_UNICODE) {
        const Uint16 *text16 = (const Uint16 *) text;
        utf8_alloc = SDL_stack_alloc(Uint8, UCS2_to_UTF8_len(text16));
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        UCS2_to_UTF8(text16, utf8_alloc);
        text_cpy = (char *)utf8_alloc;
    } else {
        /* Use a copy anyway */
        size_t str_len = SDL_strlen(text);
        utf8_alloc = SDL_stack_alloc(Uint8, str_len + 1);
        if (utf8_alloc == NULL) {
            SDL_OutOfMemory();
            goto failure;
        }
        SDL_memcpy(utf8_alloc, text, str_len + 1);
        text_cpy = (char *)utf8_alloc;
    }

#if TTF_USE_SDF
    /* Invalid cache if we were using SDF */
    if (render_mode != RENDER_BLENDED) {
        if (font->render_sdf) {
            font->render_sdf = 0;
            Flush_Cache(font);
        }
    }
#endif

    /* Get the dimensions of the text surface */
    if ((TTF_SizeUTF8(font, text_cpy, &width, &height) < 0) || !width) {
        TTF_SetError("Text has zero width");
        goto failure;
    }

    /* wrapLength is unsigned, but don't allow negative values */
    if ((int)wrapLength < 0) {
        TTF_SetError("Invalid parameter 'wrapLength'");
        goto failure;
    }

    numLines = 1;

    if (*text_cpy) {
        int maxNumLines = 0;
        size_t textlen = SDL_strlen(text_cpy);
        numLines = 0;

        do {
            int extent = 0, max_count = 0, char_count = 0;
            size_t save_textlen = (size_t)(-1);
            char *save_text  = NULL;

            if (numLines >= maxNumLines) {
                char **saved = strLines;
                if (wrapLength == 0) {
                    maxNumLines += 32;
                } else {
                    maxNumLines += (width / wrapLength) + 1;
                }
                strLines = (char **)SDL_realloc(strLines, maxNumLines * sizeof (*strLines));
                if (strLines == NULL) {
                    strLines = saved;
                    SDL_OutOfMemory();
                    goto failure;
                }
            }

            strLines[numLines++] = text_cpy;

            if (TTF_MeasureUTF8(font, text_cpy, wrapLength, &extent, &max_count) < 0) {
                TTF_SetError("Error measure text");
                goto failure;
            }

            if (wrapLength != 0) {
                if (max_count == 0) {
                    max_count = 1;
                }
            }

            while (textlen > 0) {
                int inc = 0;
                int is_delim;
                Uint32 c = UTF8_getch(text_cpy, textlen, &inc);
                text_cpy += inc;
                textlen -= inc;

                if (c == UNICODE_BOM_NATIVE || c == UNICODE_BOM_SWAPPED) {
                    continue;
                }

                char_count += 1;

                /* With wrapLength == 0, normal text rendering but newline aware */
                is_delim = (wrapLength > 0) ?  CharacterIsDelimiter(c) : CharacterIsNewLine(c);

                /* Record last delimiter position */
                if (is_delim) {
                    save_textlen = textlen;
                    save_text = text_cpy;
                    /* Break, if new line */
                    if (c == '\n' || c == '\r') {
                        *(text_cpy - 1) = '\0';
                        break;
                    }
                }

                /* Break, if reach the limit */
                if (char_count == max_count) {
                    break;
                }
            }

            /* Cut at last delimiter/new lines, otherwise in the middle of the word */
            if (save_text && textlen) {
                text_cpy = save_text;
                textlen = save_textlen;
            }
        } while (textlen > 0);
    }

    lineskip = TTF_FontLineSkip(font);
    rowHeight = SDL_max(height, lineskip);

    if (wrapLength == 0) {
        /* Find the max of all line lengths */
        if (numLines > 1) {
            width = 0;
            for (i = 0; i < numLines; i++) {
                char save_c = 0;
                int w, h;

                /* Add end-of-line */
                if (strLines) {
                    text = strLines[i];
                    if (i + 1 < numLines) {
                        save_c = strLines[i + 1][0];
                        strLines[i + 1][0] = '\0';
                    }
                }

                if (TTF_SizeUTF8(font, text, &w, &h) == 0) {
                    width = SDL_max(w, width);
                }

                /* Remove end-of-line */
                if (strLines) {
                    if (i + 1 < numLines) {
                        strLines[i + 1][0] = save_c;
                    }
                }
            }
            /* In case there are all newlines */
            width = SDL_max(width, 1);
        }
    } else {
        if (numLines <= 1 && font->horizontal_align == TTF_WRAPPED_ALIGN_LEFT) {
            /* Don't go above wrapLength if you have only 1 line which hasn't been cut */
            width = SDL_min((int)wrapLength, width);
        } else {
            width = wrapLength;
        }
    }
    height = rowHeight + lineskip * (numLines - 1);

    /* Support alpha blending */
    fg.a = fg.a ? fg.a : SDL_ALPHA_OPAQUE;
    bg.a = bg.a ? bg.a : SDL_ALPHA_OPAQUE;

    /* Create surface for rendering */
    if (render_mode == RENDER_SOLID) {
        textbuf = Create_Surface_Solid(width, height, fg, &color);
    } else if (render_mode == RENDER_SHADED) {
        textbuf = Create_Surface_Shaded(width, height, fg, bg, &color);
    } else if (render_mode == RENDER_BLENDED) {
        textbuf = Create_Surface_Blended(width, height, fg, &color);
    } else { /* render_mode == RENDER_LCD */
        textbuf = Create_Surface_LCD(width, height, fg, bg, &color);
    }

    if (textbuf == NULL) {
        goto failure;
    }

    /* Render each line */
    for (i = 0; i < numLines; i++) {
        int xstart, ystart, line_width, xoffset;
        char save_c = 0;

        /* Add end-of-line */
        if (strLines) {
            text = strLines[i];
            if (i + 1 < numLines) {
                save_c = strLines[i + 1][0];
                strLines[i + 1][0] = '\0';
            }
        }

        /* Initialize xstart, ystart and compute positions */
        if (TTF_Size_Internal(font, text, STR_UTF8, &line_width, NULL, &xstart, &ystart, NO_MEASUREMENT) < 0) {
            goto failure;
        }

        /* Move to i-th line */
        ystart += i * lineskip;

        /* Control left/right/center align of each bit of text */
        if (font->horizontal_align == TTF_WRAPPED_ALIGN_RIGHT) {
            xoffset = width - line_width;
        } else if (font->horizontal_align == TTF_WRAPPED_ALIGN_CENTER) {
            xoffset = width / 2 - line_width / 2;
        } else {
            xoffset = 0;
        }
        xoffset = SDL_max(0, xoffset);

        /* Render one text line to textbuf at (xstart, ystart) */
        if (Render_Line(render_mode, font->render_subpixel, font, textbuf, xstart + xoffset, ystart, fg) < 0) {
            goto failure;
        }

        /* Apply underline or strikethrough style, if needed */
        if (TTF_HANDLE_STYLE_UNDERLINE(font)) {
            Draw_Line(font, textbuf, xoffset, ystart + font->underline_top_row, line_width, font->line_thickness, color, render_mode);
        }

        if (TTF_HANDLE_STYLE_STRIKETHROUGH(font)) {
            Draw_Line(font, textbuf, xoffset, ystart + font->strikethrough_top_row, line_width, font->line_thickness, color, render_mode);
        }
        /* Remove end-of-line */
        if (strLines) {
            if (i + 1 < numLines) {
                strLines[i + 1][0] = save_c;
            }
        }
    }

    if (strLines) {
        SDL_free(strLines);
    }
    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return textbuf;
failure:
    if (textbuf) {
        SDL_FreeSurface(textbuf);
    }
    if (strLines) {
        SDL_free(strLines);
    }
    if (utf8_alloc) {
        SDL_stack_free(utf8_alloc);
    }
    return NULL;
}

SDL_Surface* TTF_RenderText_Solid_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_TEXT, fg, fg /* unused */, wrapLength, RENDER_SOLID);
}

SDL_Surface* TTF_RenderUTF8_Solid_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_UTF8, fg, fg /* unused */, wrapLength, RENDER_SOLID);
}

SDL_Surface* TTF_RenderUNICODE_Solid_Wrapped(TTF_Font *font, const Uint16 *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, (const char *)text, STR_UNICODE, fg, fg /* unused */, wrapLength, RENDER_SOLID);
}

SDL_Surface* TTF_RenderText_Shaded_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_TEXT, fg, bg, wrapLength, RENDER_SHADED);
}

SDL_Surface* TTF_RenderUTF8_Shaded_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_UTF8, fg, bg, wrapLength, RENDER_SHADED);
}

SDL_Surface* TTF_RenderUNICODE_Shaded_Wrapped(TTF_Font *font, const Uint16 *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, (const char *)text, STR_UNICODE, fg, bg, wrapLength, RENDER_SHADED);
}

SDL_Surface* TTF_RenderText_Blended_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_TEXT, fg, fg /* unused */, wrapLength, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderUTF8_Blended_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_UTF8, fg, fg /* unused */, wrapLength, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderUNICODE_Blended_Wrapped(TTF_Font *font, const Uint16 *text, SDL_Color fg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, (const char *)text, STR_UNICODE, fg, fg /* unused */, wrapLength, RENDER_BLENDED);
}

SDL_Surface* TTF_RenderText_LCD_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_TEXT, fg, bg, wrapLength, RENDER_LCD);
}

SDL_Surface* TTF_RenderUTF8_LCD_Wrapped(TTF_Font *font, const char *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, text, STR_UTF8, fg, bg, wrapLength, RENDER_LCD);
}

SDL_Surface* TTF_RenderUNICODE_LCD_Wrapped(TTF_Font *font, const Uint16 *text, SDL_Color fg, SDL_Color bg, Uint32 wrapLength)
{
    return TTF_Render_Wrapped_Internal(font, (const char *)text, STR_UNICODE, fg, bg, wrapLength, RENDER_LCD);
}

SDL_Surface* TTF_RenderGlyph_Blended(TTF_Font *font, Uint16 ch, SDL_Color fg)
{
    return TTF_RenderGlyph32_Blended(font, ch, fg);
}

SDL_Surface* TTF_RenderGlyph32_Blended(TTF_Font *font, Uint32 ch, SDL_Color fg)
{
    Uint8 utf8[7];

    TTF_CHECK_POINTER(font, NULL);

    if (!Char_to_UTF8(ch, utf8)) {
        return NULL;
    }

    return TTF_RenderUTF8_Blended(font, (char *)utf8, fg);
}

void TTF_SetFontStyle(TTF_Font *font, int style)
{
    int prev_style;
    long face_style;

    TTF_CHECK_POINTER(font,);

    prev_style = font->style;
    face_style = font->face->style_flags;

    /* Don't add a style if already in the font, SDL_ttf doesn't need to handle them */
    if (face_style & FT_STYLE_FLAG_BOLD) {
        style &= ~TTF_STYLE_BOLD;
    }
    if (face_style & FT_STYLE_FLAG_ITALIC) {
        style &= ~TTF_STYLE_ITALIC;
    }

    font->style = style;

    TTF_initFontMetrics(font);

    /* Flush the cache if the style has changed.
     * Ignore styles which do not impact glyph drawning. */
    if ((font->style | TTF_STYLE_NO_GLYPH_CHANGE) != (prev_style | TTF_STYLE_NO_GLYPH_CHANGE)) {
        Flush_Cache(font);
    }
}

int TTF_GetFontStyle(const TTF_Font *font)
{
    int style;
    long face_style;

    TTF_CHECK_POINTER(font, -1);

    style = font->style;
    face_style = font->face->style_flags;

    /* Add the style already in the font */
    if (face_style & FT_STYLE_FLAG_BOLD) {
        style |= TTF_STYLE_BOLD;
    }
    if (face_style & FT_STYLE_FLAG_ITALIC) {
        style |= TTF_STYLE_ITALIC;
    }

    return style;
}

void TTF_SetFontOutline(TTF_Font *font, int outline)
{
    TTF_CHECK_POINTER(font,);

    font->outline_val = SDL_max(0, outline);
    TTF_initFontMetrics(font);
    Flush_Cache(font);
}

int TTF_GetFontOutline(const TTF_Font *font)
{
    TTF_CHECK_POINTER(font, -1);

    return font->outline_val;
}

void TTF_SetFontHinting(TTF_Font *font, int hinting)
{
    TTF_CHECK_POINTER(font,);

    if (hinting == TTF_HINTING_LIGHT || hinting == TTF_HINTING_LIGHT_SUBPIXEL) {
        font->ft_load_target = FT_LOAD_TARGET_LIGHT;
    } else if (hinting == TTF_HINTING_MONO) {
        font->ft_load_target = FT_LOAD_TARGET_MONO;
    } else if (hinting == TTF_HINTING_NONE) {
        font->ft_load_target = FT_LOAD_NO_HINTING;
    } else {
        font->ft_load_target = FT_LOAD_TARGET_NORMAL;
    }

    font->render_subpixel = (hinting == TTF_HINTING_LIGHT_SUBPIXEL) ? 1 : 0;
#if TTF_USE_HARFBUZZ
    /* update flag for HB */
    hb_ft_font_set_load_flags(font->hb_font, FT_LOAD_DEFAULT | font->ft_load_target);
#endif

    Flush_Cache(font);
}

int TTF_GetFontHinting(const TTF_Font *font)
{
    TTF_CHECK_POINTER(font, -1);

    if (font->ft_load_target == FT_LOAD_TARGET_LIGHT) {
        if (font->render_subpixel == 0) {
            return TTF_HINTING_LIGHT;
        } else {
            return TTF_HINTING_LIGHT_SUBPIXEL;
        }
    } else if (font->ft_load_target == FT_LOAD_TARGET_MONO) {
        return TTF_HINTING_MONO;
    } else if (font->ft_load_target == FT_LOAD_NO_HINTING) {
        return TTF_HINTING_NONE;
    }
    return TTF_HINTING_NORMAL;
}

int TTF_SetFontSDF(TTF_Font *font, SDL_bool on_off)
{
    TTF_CHECK_POINTER(font, -1);
#if TTF_USE_SDF
    font->render_sdf = on_off;
    Flush_Cache(font);
    return 0;
#else
    TTF_SetError("SDL_ttf compiled without SDF support");
    return -1;
#endif
}

SDL_bool TTF_GetFontSDF(const TTF_Font *font)
{
    TTF_CHECK_POINTER(font, SDL_FALSE);
    return font->render_sdf;
}

void TTF_SetFontWrappedAlign(TTF_Font *font, int align)
{
    TTF_CHECK_POINTER(font,);

    /* input not checked, unknown values assumed to be TTF_WRAPPED_ALIGN_LEFT */
    if (align == TTF_WRAPPED_ALIGN_CENTER) {
        font->horizontal_align = TTF_WRAPPED_ALIGN_CENTER;
    } else if (align == TTF_WRAPPED_ALIGN_RIGHT) {
        font->horizontal_align = TTF_WRAPPED_ALIGN_RIGHT;
    } else {
        font->horizontal_align = TTF_WRAPPED_ALIGN_LEFT;
    }
}

int TTF_GetFontWrappedAlign(const TTF_Font *font)
{
    TTF_CHECK_POINTER(font,-1);
    return font->horizontal_align;
}

void TTF_Quit(void)
{
    if (TTF_initialized) {
        if (--TTF_initialized == 0) {
            FT_Done_FreeType(library);
            library = NULL;
        }
    }
}

int TTF_WasInit(void)
{
    return TTF_initialized;
}

/* don't use this function. It's just here for binary compatibility. */
int TTF_GetFontKerningSize(TTF_Font *font, int prev_index, int index)
{
    FT_Vector delta;

    TTF_CHECK_POINTER(font, -1);

    FT_Get_Kerning(font->face, (FT_UInt)prev_index, (FT_UInt)index, FT_KERNING_DEFAULT, &delta);
    return (int)(delta.x >> 6);
}

int TTF_GetFontKerningSizeGlyphs(TTF_Font *font, Uint16 previous_ch, Uint16 ch)
{
    return TTF_GetFontKerningSizeGlyphs32(font, previous_ch, ch);
}

int TTF_GetFontKerningSizeGlyphs32(TTF_Font *font, Uint32 previous_ch, Uint32 ch)
{
    FT_Error error;
    c_glyph *prev_glyph, *glyph;
    FT_Vector delta;

    TTF_CHECK_POINTER(font, -1);

    if (ch == UNICODE_BOM_NATIVE || ch == UNICODE_BOM_SWAPPED) {
        return 0;
    }

    if (previous_ch == UNICODE_BOM_NATIVE || previous_ch == UNICODE_BOM_SWAPPED) {
        return 0;
    }

    if (Find_GlyphMetrics(font, ch, &glyph) < 0) {
        return -1;
    }

    if (Find_GlyphMetrics(font, previous_ch, &prev_glyph) < 0) {
        return -1;
    }

    error = FT_Get_Kerning(font->face, prev_glyph->index, glyph->index, FT_KERNING_DEFAULT, &delta);
    if (error) {
        TTF_SetFTError("Couldn't get glyph kerning", error);
        return -1;
    }
    return (int)(delta.x >> 6);
}

/* vi: set ts=4 sw=4 expandtab: */
