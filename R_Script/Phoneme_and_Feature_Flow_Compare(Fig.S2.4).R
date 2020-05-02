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
unit_per_Row <- 8
row_per_Page <- 2

for (epoch in epoch_List)
{
  for (identifier in identifier_List)
  {
    work_Dir <- file.path(base_Dir, paste(identifier, '.', 'IDX', index, sep=''), 'Hidden')
    
    for (flow_Type in c("Phone", "Feature"))
    {
      if (!dir.exists(file.path(work_Dir,'Flow', 'PNG.Compare')))
      {
        dir.create(file.path(work_Dir,'Flow', 'PNG.Compare'))
      }
      
      flow_Data_List <- list()
      
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
        
        if (unit_Index == 0) {
          if (flow_Type == 'Phone') { row_Name.Phone <- rownames(flow_Data) }
          if (flow_Type == 'Feature') { row_Name.Feature <- rownames(flow_Data) }
        }
        
        flow_Data$row_Name <- rownames(flow_Data)
        flow_Data.Melt <- melt(as.data.frame(flow_Data), id.vars = "row_Name", variable.name = 'time')
        flow_Data.Melt$unit_Index <- unit_Index
        
        flow_Data_List[[length(flow_Data_List) + 1]] <- flow_Data.Melt
      }
      if (flow_Type == 'Phone') { flow_Data.Phone <- do.call(rbind, flow_Data_List) }
      else if (flow_Type == 'Feature') { flow_Data.Feature <- do.call(rbind, flow_Data_List) }
    }
    flow_Data.Phone$row_Name <- with(flow_Data.Phone, factor(row_Name, levels=rev(row_Name.Phone), ordered=TRUE))
    flow_Data.Feature$row_Name <- with(flow_Data.Feature, factor(row_Name, levels=rev(row_Name.Feature), ordered=TRUE))
    
    
    plot_List <- list()
    for(start_Unit_Index in seq(0, hidden_Unit - 1, unit_per_Row))
    {
      flow_Data.Phone.Subset <- subset(flow_Data.Phone, unit_Index %in% seq(start_Unit_Index, start_Unit_Index + unit_per_Row - 1, 1))
      
      start_Window <- min(as.numeric(as.matrix(flow_Data.Phone.Subset$time)))
      end_Window <- max(as.numeric(as.matrix(flow_Data.Phone.Subset$time)))
      
      plot_List[[length(plot_List) + 1]] <- ggplot(flow_Data.Phone.Subset, aes(x=time, y=row_Name)) +
        geom_tile(aes(fill=value)) +
        scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
        facet_grid(.~unit_Index) +
        scale_x_discrete(
          breaks = c(start_Window, seq(0, end_Window, by = 10), end_Window),
          labels = c(start_Window, seq(0, end_Window, by = 10), end_Window) * 10,
          expand=c(0,0)
        ) +
        labs(title='', x='', y='', fill='') +
        theme(
          title = element_text(size=20),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size=12),
          axis.text.y = element_text(size=12),
          axis.ticks = element_blank(),
          legend.position="none", #"right",
          legend.direction="vertical",
          legend.key.height = unit(20, "mm"),
          plot.margin=unit(c(0,0,0,0),"cm"),
          panel.grid=element_blank()
        )
      
      flow_Data.Feature.Subset <- subset(flow_Data.Feature, unit_Index %in% seq(start_Unit_Index, start_Unit_Index + unit_per_Row - 1, 1))
      
      plot_List[[length(plot_List) + 1]] <- ggplot(flow_Data.Feature.Subset, aes(x=time, y=row_Name)) +
        geom_tile(aes(fill=value)) +
        scale_fill_viridis(option="plasma", limits=c(0, 1), breaks=c(0, 1),labels=c(0, 1)) +
        facet_grid(.~unit_Index) +
        scale_x_discrete(
          breaks = c(start_Window, seq(0, end_Window, by = 10), end_Window),
          labels = c(start_Window, seq(0, end_Window, by = 10), end_Window) * 10,
          expand=c(0,0)
        ) +
        labs(title='', x='', y='', fill='') +
        theme(
          title = element_text(size=20),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.text.x = element_text(size=12),
          axis.text.y = element_text(size=12),
          axis.ticks = element_blank(),
          legend.position="none", #"right",
          legend.direction="vertical",
          legend.key.height = unit(20, "mm"),
          plot.margin=unit(c(0,0,0,0),"cm"),
          panel.grid=element_blank()
        )
      
      if((start_Unit_Index + unit_per_Row) %% (unit_per_Row * row_per_Page) == 0 || start_Unit_Index + unit_per_Row >= hidden_Unit)
      {
        if ((start_Unit_Index + unit_per_Row) %% (unit_per_Row * row_per_Page) == 0)
        {
          page_Start_Index <- start_Unit_Index - (unit_per_Row * (row_per_Page - 1))
          page_Last_Index <- start_Unit_Index + unit_per_Row - 1  
        }
        else
        {
          page_Start_Index <- hidden_Unit - (unit_per_Row) * (length(plot_List) / 2)
          page_Last_Index <- hidden_Unit - 1
        }

        ggsave(
          filename = file.path(work_Dir,'Flow', 'PNG.Compare', paste(flow_Type, '.Compare.U_', page_Start_Index, '-', page_Last_Index, '.Flow_Compare.png', sep='')),
          plot = plot_grid(plotlist=plot_List[1:(row_per_Page*2)], align = "v", ncol=1),
          device = "png",
          width = 21.6 * 3,
          height = 28 * 3,
          units = "cm",
          dpi = 300
        )
        
        plot_List <- list()
      }
    }
  }
}
