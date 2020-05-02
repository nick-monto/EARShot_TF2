library(ggplot2)
library(reshape2)
library(ggdendro)
library(grid)
library(gridExtra)
library(readr)
library(viridis)
library(cowplot)

base_Dir <- 'D:/Python_Programming/EARShot_TF2/Results'
identifier_List <- c('AGNES')
epoch_List <- c(4000)
hidden_Unit <- 512
index <- 0



for (epoch in epoch_List)
{
  for (identifier in identifier_List)
  {
    work_Dir <- file.path(base_Dir, paste(identifier, '.', 'IDX', index, sep=''), 'Hidden')
    
    #for (flow_Type in c("Phone", "Feature"))
    for (flow_Type in c("Feature"))
    { 
      if (!dir.exists(file.path(work_Dir,'Flow', flow_Type, 'PNG')))
      {
        dir.create(file.path(work_Dir,'Flow', flow_Type, 'PNG'))
      }
      
      plot_List <- list()
      for (unit_Index in seq(0, hidden_Unit - 1, 1))
      {
        flow_Data <- read_delim(
          file.path(work_Dir,'Flow', flow_Type, 'TXT', paste(flow_Type, '.U_', sprintf('%04d', unit_Index), '.I_ALL.txt', sep='')),
          delim= "\t",
          escape_double = FALSE,
          locale = locale(encoding = "UTF-8"),
          trim_ws = TRUE
          )
        flow_Data.row_Name <- as.matrix(flow_Data[1])
        flow_Data <- abs(flow_Data[,-1])
        rownames(flow_Data) <- flow_Data.row_Name
        
        
        mean_Flow_Data <- colMeans(flow_Data)
        mean_Flow_Data <- as.data.frame(mean_Flow_Data)
        colnames(mean_Flow_Data) <- c("Mean")
        mean_Flow_Data$Step <- as.numeric(rownames(mean_Flow_Data))
        
        col_Min <- min(as.numeric(colnames(flow_Data)), na.rm = TRUE)
        col_Max <- max(as.numeric(colnames(flow_Data)), na.rm = TRUE)
        
        
        flow_Data$row_Name.num <- rev(1:length(rownames(flow_Data)))
        key.flow_Data.row_Name <- data.frame(row_Name = rownames(flow_Data), row_Name.num = (1:length(rownames(flow_Data))))
        mdf <- melt(as.data.frame(flow_Data), id.vars="row_Name.num")
        mdf <- merge(mdf, key.flow_Data.row_Name, by = "row_Name.num", all.x = TRUE)
        ylabels = rev(rownames(flow_Data))
        
        plot <- ggplot(mdf, aes(x=variable, y=row_Name.num)) +
          geom_tile(aes(fill=value)) +
          scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
          scale_x_discrete(
            breaks = c(col_Min, seq(0, col_Max, by = 5), col_Max),
            labels = c(col_Min, seq(0, col_Max, by = 5), col_Max) * 10
          ) +
          scale_y_continuous(
            expand=c(0,0),
            breaks = seq(1, max(mdf$row_Name.num), by = 1),
            labels = ylabels,
            sec.axis = dup_axis()
          ) +
          labs(title=sprintf('%s flow    Unit: %s', flow_Type, unit_Index), x= 'Time (ms)', y= flow_Type, fill="") +
          theme(
            title = element_text(size=20),
            axis.title.x = element_text(size=20),
            axis.title.y = element_text(size=20),
            axis.title.y.right = element_text(size=20),
            axis.text.x = element_text(size=18),
            axis.text.y = element_text(size=18),
            axis.ticks = element_blank(),
            legend.position="right",
            legend.direction="vertical",
            legend.key.height = unit(20, "mm"),
            plot.margin=unit(c(0,0,0,0),"cm"),
            panel.grid=element_blank()
          )
        
        if (flow_Type == "Phone")
        {
          ggsave(
            filename = file.path(work_Dir,'Flow', flow_Type, 'PNG', paste(flow_Type, '.U_', sprintf('%04d', unit_Index), '.I_ALL.png', sep='')),
            plot = plot,
            device = "png",
            width = 25,
            height = 25,
            units = "cm",
            dpi = 300
          )
        }
        if (flow_Type == "Feature")
        {
          ggsave(
            filename = file.path(work_Dir,'Flow', flow_Type, 'PNG', paste(flow_Type, '.U_', sprintf('%04d', unit_Index), '.I_ALL.png', sep='')),
            plot = plot,
            device = "png",
            width = 30,
            height = 25, #10,
            units = "cm",
            dpi = 300
          )
        }
      }
    }
  }
}