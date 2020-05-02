#Dendrogram refer:
#https://stackoverflow.com/questions/6673162/reproducing-lattice-dendrogram-graph-with-ggplot2
#https://stackoverflow.com/questions/48664746/how-to-set-two-x-axis-and-two-y-axis-using-ggplot2


library(ggplot2)
library(reshape2)
library(ggdendro)
library(grid)
library(gridExtra)
library(readr)
library(viridis)

get_legend<-function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


### Set up a blank theme
theme_none <- theme(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
  axis.line = element_blank(),
  #axis.ticks.length = element_blank()
  plot.margin=unit(c(0,0,0,0),"cm"),
  panel.grid=element_blank(),
  axis.title=element_blank(),
  axis.ticks=element_blank(),
  axis.text=element_blank()
)

base_Dir <- 'D:/Python_Programming/EARShot_TF2/Results'
identifier_List <- c('AGNES')
epoch_List <- c(4000)
index <- 0
criterion_List <- round(seq(0.00, 0.50, 0.05), 2)

for (epoch in epoch_List)
{
  for (identifier in identifier_List)
  {
    work_Dir <- file.path(base_Dir, paste(identifier, '.', 'IDX', index, sep=''), 'Hidden')
    
  
    for (map_Type in c("PSI", "FSI"))
    {
      if (!dir.exists(file.path(work_Dir, "Map", map_Type, "PNG")))
      {
        dir.create(file.path(work_Dir, "Map", map_Type, "PNG"))
      }
      if (!dir.exists(file.path(work_Dir, "Map", map_Type, "PNG.NoColSort")))
      {
        dir.create(file.path(work_Dir, "Map", map_Type, "PNG.NoColSort"))
      }
    }
    
    for (criterion in criterion_List)
    {
      criterion <- format(round(criterion, 2), nsmall = 2)
      
      for (map_Type in c("PSI", "FSI"))
      {
        if (map_Type == "PSI") { row_Title="Phone" }
        if (map_Type == "FSI") { row_Title="Feature" }
        
        map_Data <- read_delim(
          file.path(work_Dir, "Map", map_Type, "TXT", paste(map_Type, '.', 'C_', criterion, '.I_ALL.txt', sep= '')),
          delim= '\t',
          escape_double = FALSE,
          locale = locale(encoding = "UTF-8"),
          trim_ws = TRUE)
        map_Data.row_Name <- as.matrix(map_Data[1])
        map_Data <- map_Data[,-1]
        map_Data <- map_Data[, colSums(map_Data != 0) > 0]
        rownames(map_Data) <- map_Data.row_Name
        
        map_Limit <- nrow(map_Data) - 1
        
        if (sum(map_Data) == 0)
        {
          next
        }
        
        x <- as.matrix(scale(map_Data))
        
        if (ncol(x) < 2) { next }
        
        dd.row <- as.dendrogram(hclust(dist(t(x))))
        row.ord <- order.dendrogram(dd.row)
        
        xx <- scale(map_Data)[, row.ord]
        xx_names <- attr(xx, "dimnames")
        xx <- map_Data[, row.ord]
        rownames(xx) <- rownames(scale(map_Data)[, row.ord])
        df <- as.data.frame(xx)
        colnames(df) <- xx_names[[2]]
        df$row_Name = xx_names[[1]]
        df$row_Name <- with(df, factor(row_Name, levels=row_Name, ordered=TRUE))
        
        mdf <- melt(df, id.vars="row_Name")
        
        key.mdf.row_Name <- data.frame(row_Name = rownames(map_Data), row_Name.num = (1:length(rownames(map_Data))))
        mdf <- merge(mdf, key.mdf.row_Name, by = "row_Name", all.x = TRUE)
        ylabels = rownames(map_Data)
        
        ddata_x <- dendro_data(dd.row)
        
        
        p1 <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
          geom_tile(aes(fill=value)) +
          scale_fill_viridis(option="plasma", limits=c(0, map_Limit), breaks=c(0, map_Limit),labels=c(0, map_Limit)) +
          scale_y_continuous(
            trans = "reverse",
            expand=c(0,0),
            breaks = seq(1, max(mdf$row_Name.num), by = 1),
            labels = ylabels,
            sec.axis = dup_axis()
          ) +
          labs(y=row_Title, fill="") +
          theme(
            axis.title.x = element_blank(),
            axis.title.y = element_text(size=20),
            axis.title.y.right = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_text(size=15),
            axis.ticks = element_blank(),
            legend.position=c(1.07, -0.15),
            legend.direction="horizontal",
            legend.text = element_text(size=20),
            plot.margin=unit(c(0,0,0,0),"cm"),
            panel.grid=element_blank()
          )
        
        # Dendrogram 1
        p2 <- ggplot(segment(ddata_x)) +
          geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
          theme_none + theme(axis.title.x=element_blank()) + scale_y_reverse()
        
        if (map_Type == "PSI")
        {
          png(
            file.path(work_Dir, "Map", map_Type, "PNG.NoColSort", paste(map_Type, '.', 'C_', criterion, '.I_ALL.png', sep= '')),
            width = 350,
            height = 250,
            res =300,
            units = "mm"
            )
          grid.newpage()
          print(p2, vp=viewport(0.918, 0.2, x=0.46, y=0.095))
          print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
          dev.off()
        }
        if (map_Type == "FSI")
        {
          png(
            file.path(work_Dir, "Map", map_Type, "PNG.NoColSort", paste(map_Type, '.', 'C_', criterion, '.I_ALL.png', sep= '')),
            width = 500,
            height = 200,
            res =300,
            units = "mm"
            )
          grid.newpage()
          print(p2, vp=viewport(0.79, 0.2, x=0.457, y=0.095))
          print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
          dev.off()
        }
        
        
        x <- as.matrix(scale(map_Data))
        dd.col <- as.dendrogram(hclust(dist(x)))
        col.ord <- order.dendrogram(dd.col)
        
        dd.row <- as.dendrogram(hclust(dist(t(x))))
        row.ord <- order.dendrogram(dd.row)
        
        xx <- scale(map_Data)[col.ord, row.ord]
        xx_names <- attr(xx, "dimnames")
        xx <- map_Data[col.ord, row.ord]
        rownames(xx) <- rownames(scale(map_Data)[col.ord, row.ord])
        df <- as.data.frame(xx)
        colnames(df) <- xx_names[[2]]
        df$row_Name = xx_names[[1]]
        df$row_Name <- with(df, factor(row_Name, levels=row_Name, ordered=TRUE))
        
        mdf <- melt(df, id.vars="row_Name")
        
        key.mdf.row_Name <- data.frame(row_Name = rownames(map_Data)[col.ord], row_Name.num = (1:length(rownames(map_Data))))
        mdf <- merge(mdf, key.mdf.row_Name, by = "row_Name", all.x = TRUE)
        ylabels = rownames(map_Data)[col.ord]
        
        ddata_x <- dendro_data(dd.row)
        ddata_y <- dendro_data(dd.col)
        
        
        
        ### Create plot components ###
        # Heatmap
        p1 <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
          geom_tile(aes(fill=value)) +
          scale_fill_viridis(option="plasma", limits=c(0, map_Limit), breaks=c(0, map_Limit),labels=c(0, map_Limit)) +
          scale_y_continuous(
            expand=c(0,0),
            breaks = seq(1, max(mdf$row_Name.num), by = 1),
            labels = ylabels,
            sec.axis = dup_axis()
          ) +
          labs(y=row_Title, fill="") +
          theme(
            axis.title.x = element_blank(),
            axis.title.y = element_text(size=20),
            axis.title.y.right = element_blank(),
            axis.text.x = element_blank(),
            axis.text.y = element_text(size=15),
            axis.ticks = element_blank(),
            legend.position=c(1.07, -0.15),
            legend.direction="horizontal",
            legend.text = element_text(size=20),
            plot.margin=unit(c(0,0,0,0),"cm"),
            panel.grid=element_blank()
          )
        
        
        
        # Dendrogram 1
        p2 <- ggplot(segment(ddata_x)) +
          geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
          theme_none + theme(axis.title.x=element_blank()) + scale_y_reverse()
        
        # Dendrogram 2
        p3 <- ggplot(segment(ddata_y)) +
          geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
          coord_flip() +
          theme_none +
          theme(axis.title.x=element_blank())
        geom_text(data=label(ddata_y), aes(label=label, x=x, y=0), hjust=0.5,size=3)
        
        ### Draw graphic ###
        if (map_Type == "PSI")
        { 
          png(
            file.path(work_Dir, "Map", map_Type, "PNG", paste(map_Type, '.', 'C_', criterion, '.I_ALL.png', sep= '')),
            width = 350,
            height = 250,
            res =300,
            units = "mm"
            )
          grid.newpage()
          print(p2, vp=viewport(0.918, 0.2, x=0.46, y=0.095))
          print(p3, vp=viewport(0.1, 0.855, x=0.95, y=0.59))
          print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
          dev.off()
        }
        if (map_Type == "FSI")
        {
          png(
            file.path(work_Dir, "Map", map_Type, "PNG", paste(map_Type, '.', 'C_', criterion, '.I_ALL.png', sep= '')),
            width = 500,
            height = 200,
            res =300,
            units = "mm"
            )
          grid.newpage()
          print(p2, vp=viewport(0.79, 0.2, x=0.457, y=0.095))
          print(p3, vp=viewport(0.1, 0.855, x=0.95, y=0.59))
          print(p1, vp=viewport(0.9, 0.8, x=0.45, y=0.59))
          
          dev.off()
        }
      }
    }
  }
}
