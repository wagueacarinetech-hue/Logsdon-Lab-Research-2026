# ─────────────────────────────────────────────────────────────────────────────
# MUC1 repeat architecture plot
#
# Plots RepeatMasker annotations across all extracted MUC1 haplotype assemblies.
# Each row is one haplotype; each colored segment is a repeat element drawn at
# its position in the extracted interval and colored by repeat class.
#
# Two things to look for: interspersed elements (LINE, SINE, LTR) should sit at
# consistent positions across rows, which is a check on assembly quality. The
# MUC1 VNTR appears as a Simple repeat block whose width varies with copy number.
#
# Adapted from the LPA version written by Saron Akalu, with the input paths and
# filename parsing changed for the MUC1 locus.
#
# Input:  RepeatMasker .out files, one per haplotype
# Output: PDF, one row per haplotype
# ─────────────────────────────────────────────────────────────────────────────



# Plot RepeatMasker annotations across all extracted MUC1 assemblies
# Adapted from the LPA version

library(data.table)
library(dplyr)
library(ggplot2)
library(stringr)
library(purrr)


# Input directory containing the RepeatMasker .out files
repeatmasker_dir <- "~/Desktop/repeatmasker_out"

# Output figure
out_pdf <- "~/Desktop/MUC1_repeatmasker_all_assemblies.pdf"


# Find all RepeatMasker .out files
rm_files <- list.files(
  path.expand(repeatmasker_dir),
  pattern = "\\.out$",
  full.names = TRUE
)

if (length(rm_files) == 0) {
  stop("No RepeatMasker .out files were found in: ", repeatmasker_dir)
}

message("Found ", length(rm_files), " RepeatMasker output files")


# Read one RepeatMasker .out file
read_repeatmasker_out <- function(file_path) {

  df <- fread(
    file_path,
    skip = 3,
    fill = TRUE,
    header = FALSE,
    quote = "",
    data.table = FALSE
  )

  if (nrow(df) == 0) {
    return(data.frame())
  }

  df <- df[, 1:15]

  colnames(df) <- c(
    "score",
    "percent_divergence",
    "percent_deletions",
    "percent_insertions",
    "sequence_name",
    "start",
    "end",
    "bases_left",
    "strand",
    "repeat_name",
    "repeat_class_family",
    "repeat_start",
    "repeat_end",
    "repeat_bases_left",
    "repeat_id"
  )

  # Use the RepeatMasker output filename as the assembly label
  # the MUC1 padding suffix is stripped so row labels stay readable
  assembly_name <- basename(file_path) %>%
    str_remove("\\.out$") %>%
    str_remove("\\.fa$") %>%
    str_remove("\\.fasta$") %>%
    str_remove("\\.MUC1_10kb_padding\\.oriented$")

  df <- df %>%
    mutate(
      assembly = assembly_name,
      start = as.numeric(start),
      end = as.numeric(end)
    )

  return(df)
}


# Read and combine RepeatMasker annotations from all assemblies
rm_data <- map_dfr(
  rm_files,
  read_repeatmasker_out
)

if (nrow(rm_data) == 0) {
  stop("No usable RepeatMasker annotations were found.")
}

message("Parsed ", nrow(rm_data), " repeat annotations")


# Split the RepeatMasker class/family field
rm_data <- rm_data %>%
  mutate(
    repeat_class = str_split_fixed(repeat_class_family, "/", 2)[, 1],
    repeat_family = str_split_fixed(repeat_class_family, "/", 2)[, 2]
  )


# Collapse RepeatMasker annotations into broader repeat classes
rm_data <- rm_data %>%
  mutate(
    repeat_group = case_when(
      repeat_class == "LINE" ~ "LINE",
      repeat_class == "SINE" ~ "SINE",
      repeat_class == "LTR" ~ "LTR",
      repeat_class == "DNA" ~ "DNA",
      repeat_class == "RC" ~ "RC/Helitron",
      repeat_class == "Simple_repeat" ~ "Simple repeat",
      repeat_class == "Low_complexity" ~ "Low complexity",
      repeat_class == "Satellite" ~ "Satellite",
      repeat_class == "Retroposon" ~ "Retroposon",
      repeat_class == "rRNA" ~ "rRNA",
      repeat_class == "scRNA" ~ "scRNA",
      repeat_class == "snRNA" ~ "snRNA",
      repeat_class == "srpRNA" ~ "srpRNA",
      repeat_class == "tRNA" ~ "tRNA",
      TRUE ~ "Other"
    )
  )


# Set repeat-class order
repeat_order <- c(
  "LINE", "SINE", "LTR", "DNA", "RC/Helitron",
  "Simple repeat", "Low complexity", "Satellite", "Retroposon",
  "rRNA", "scRNA", "snRNA", "srpRNA", "tRNA", "Other"
)

rm_data$repeat_group <- factor(rm_data$repeat_group, levels = repeat_order)


# Plot assemblies alphabetically
assembly_order <- sort(unique(rm_data$assembly))
rm_data$assembly <- factor(rm_data$assembly, levels = rev(assembly_order))


# Repeat-class colors
repeat_colors <- c(
  "LINE" = "#E571AB",
  "SINE" = "#47B1B5",
  "LTR" = "#E1665E",
  "DNA" = "#8F90C6",
  "RC/Helitron" = "#CF9334",
  "Simple repeat" = "#508F52",
  "Low complexity" = "#BFBFBF",
  "Satellite" = "#AD8C2A",
  "Retroposon" = "#55ACE0",
  "rRNA" = "#A5A437",
  "scRNA" = "#54BC83",
  "snRNA" = "#CD76B0",
  "srpRNA" = "#55BDC3",
  "tRNA" = "#89AC40",
  "Other" = "#777777"
)


# Check the number of annotations in each repeat class for each assembly
repeat_summary <- rm_data %>%
  count(assembly, repeat_group, name = "number_of_annotations") %>%
  arrange(assembly, desc(number_of_annotations))

print(head(repeat_summary, 30))


# Draw LINE annotations after the other repeat classes
other_repeats <- rm_data %>% filter(repeat_group != "LINE")
line_repeats  <- rm_data %>% filter(repeat_group == "LINE")


# Plot the RepeatMasker annotations
repeat_plot <- ggplot() +

  geom_segment(
    data = other_repeats,
    aes(x = start / 1000, xend = end / 1000,
        y = assembly, yend = assembly,
        color = repeat_group),
    linewidth = 5,
    lineend = "butt"
  ) +

  geom_segment(
    data = line_repeats,
    aes(x = start / 1000, xend = end / 1000,
        y = assembly, yend = assembly,
        color = repeat_group),
    linewidth = 5,
    lineend = "butt"
  ) +

  scale_color_manual(values = repeat_colors, drop = FALSE) +

  labs(
    title = "RepeatMasker annotations across MUC1 assemblies",
    x = "Position within extracted MUC1 sequence (kb)",
    y = NULL,
    color = "Repeat class"
  ) +

  theme_classic(base_size = 12) +

  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "bottom",
    legend.key.width = grid::unit(1.1, "cm"),
    plot.title = element_text(face = "bold")
  ) +

  guides(
    color = guide_legend(
      override.aes = list(linewidth = 5),
      nrow = 2,
      byrow = TRUE
    )
  )


# Set figure height based on the number of assemblies
number_of_assemblies <- n_distinct(rm_data$assembly)
plot_height <- max(5, 0.35 * number_of_assemblies + 2.5)


# Save the plot
ggsave(
  filename = path.expand(out_pdf),
  plot = repeat_plot,
  width = 14,
  height = plot_height,
  limitsize = FALSE
)



message("Saved plot to: ", out_pdf)

# ─── Zoomed panel: the VNTR region ───────────────────────────────────────────
# Cropping to the array region makes RepeatMasker's annotations there visible.
# Note this panel does NOT show copy-number variation: RepeatMasker annotates
# only fragments of the VNTR, not the whole array, so Simple repeat width stays
# roughly constant regardless of copy number. TRF measures the array — see the
# next panel.

ZOOM_START <- 55000
ZOOM_END   <- 72000

copy_numbers <- fread("~/Desktop/MUC1_copy_numbers.tsv")

# rebuild the haplotype label to match the RepeatMasker filenames
copy_numbers <- copy_numbers %>%
  mutate(assembly = paste0(sample, "_", haplotype)) %>%
  select(assembly, copy_number)

rm_zoom <- rm_data %>%
  mutate(assembly = as.character(assembly)) %>%
  inner_join(copy_numbers, by = "assembly") %>%
  filter(end >= ZOOM_START, start <= ZOOM_END)

message("Zoom panel: ", n_distinct(rm_zoom$assembly), " haplotypes matched to copy numbers")

# order rows by copy number, lowest at the bottom
zoom_order <- copy_numbers %>%
  filter(assembly %in% rm_zoom$assembly) %>%
  arrange(copy_number) %>%
  pull(assembly)

rm_zoom$assembly <- factor(rm_zoom$assembly, levels = zoom_order)

zoom_plot <- ggplot(rm_zoom) +

  geom_segment(
    aes(x = start / 1000, xend = end / 1000,
        y = assembly, yend = assembly,
        color = repeat_group),
    linewidth = 4,
    lineend = "butt"
  ) +

  scale_color_manual(values = repeat_colors, drop = FALSE) +

  coord_cartesian(xlim = c(ZOOM_START / 1000, ZOOM_END / 1000)) +

  labs(
    title = "RepeatMasker annotations across the MUC1 VNTR region",
    subtitle = "Rows ordered by copy number; Simple repeat annotation does not scale with array length",
    x = "Position within extracted MUC1 sequence (kb)",
    y = NULL,
    color = "Repeat class"
  ) +

  theme_classic(base_size = 12) +

  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 5),
    legend.position = "bottom",
    legend.key.width = grid::unit(1.1, "cm"),
    plot.title = element_text(face = "bold")
  ) +

  guides(color = guide_legend(override.aes = list(linewidth = 5), nrow = 2, byrow = TRUE))

ggsave(
  filename = path.expand("~/Desktop/MUC1_repeatmasker_VNTR_zoom.pdf"),
  plot = zoom_plot,
  width = 10,
  height = plot_height,
  limitsize = FALSE
)

message("Saved zoom to: ~/Desktop/MUC1_repeatmasker_VNTR_zoom.pdf")


# ─── TRF array intervals, ordered by copy number ─────────────────────────────
# RepeatMasker annotates only fragments of the VNTR, so its Simple repeat blocks
# don't scale with array length. TRF measures the array directly, so plotting its
# intervals gives a true picture of the copy-number variation. Records are filtered
# the same way as in the copy-number analysis: 55-65bp period window, boundary
# elements dropped.

TRF_DIR <- "~/Desktop/trf"

read_trf_dat <- function(file_path) {
  lines <- readLines(file_path, warn = FALSE)
  fields <- strsplit(lines, "\\s+")
  keep <- vapply(fields, function(f) length(f) >= 14 && grepl("^[0-9]+$", f[1]), logical(1))
  if (!any(keep)) return(data.frame())

  f <- fields[keep]
  df <- data.frame(
    start        = as.numeric(vapply(f, `[`, character(1), 1)),
    end          = as.numeric(vapply(f, `[`, character(1), 2)),
    period       = as.numeric(vapply(f, `[`, character(1), 3)),
    trf_copies   = as.numeric(vapply(f, `[`, character(1), 4)),
    pct_matches  = as.numeric(vapply(f, `[`, character(1), 6)),
    stringsAsFactors = FALSE
  )

  df$assembly <- basename(file_path) %>% str_remove("\\.trf\\.dat$")

  # same filters as the copy-number analysis
  df %>%
    filter(period >= 55, period <= 65) %>%
    filter(!(trf_copies < 4 & pct_matches < 95))
}

trf_files <- list.files(path.expand(TRF_DIR), pattern = "\\.dat$", full.names = TRUE)
trf_data <- map_dfr(trf_files, read_trf_dat)

message("TRF: ", nrow(trf_data), " array records across ",
        n_distinct(trf_data$assembly), " haplotypes")

trf_data <- trf_data %>%
  inner_join(copy_numbers, by = "assembly")

trf_order <- copy_numbers %>%
  filter(assembly %in% trf_data$assembly) %>%
  arrange(copy_number) %>%
  pull(assembly)

trf_data$assembly <- factor(trf_data$assembly, levels = trf_order)

# sanity check the ordering — lowest copy number should be the first factor
# level, which ggplot draws at the bottom of the panel
message("Bottom row: ", trf_order[1],
        " (", round(copy_numbers$copy_number[copy_numbers$assembly == trf_order[1]], 2), " copies)")
message("Top row: ", trf_order[length(trf_order)],
        " (", round(copy_numbers$copy_number[copy_numbers$assembly == trf_order[length(trf_order)]], 2), " copies)")

trf_plot <- ggplot(trf_data) +

  geom_segment(
    aes(x = start / 1000, xend = end / 1000,
        y = assembly, yend = assembly,
        color = copy_number),
    linewidth = 4,
    lineend = "butt"
  ) +

  scale_color_viridis_c(option = "plasma", name = "Copy number\n(Jellyfish)") +

  # cropped to the array region so bar length differences are visible, not just
  # color — 20 vs 80 copies is only ~3.6kb, invisible on the full locus axis
  coord_cartesian(xlim = c(60, 70)) +

  labs(
    title = "MUC1 VNTR array extent measured by TRF",
    subtitle = "Bar extent from TRF array intervals; color from Jellyfish copy number. Ordered by copy number, lowest at bottom.",
    y = NULL
  ) +

  theme_classic(base_size = 12) +

  theme(
    axis.line.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 5),
    legend.position = "right",
    plot.title = element_text(face = "bold")
  )

ggsave(
  filename = path.expand("~/Desktop/MUC1_TRF_array_extent.pdf"),
  plot = trf_plot,
  width = 10,
  height = plot_height,
  limitsize = FALSE
)

message("Saved TRF array plot to: ~/Desktop/MUC1_TRF_array_extent.pdf")