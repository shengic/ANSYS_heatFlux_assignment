

Private Sub Frame6_Click()

End Sub

Private Sub Label46_Click()

End Sub

Private Sub Label77_Click()

End Sub

Private Sub TextBox1_Change()

End Sub

Private Sub updateGeometryButton_Click()
' determine the centroid point of a heatflux surface to ansysHFluxElement.elementNodes
   Call storeAnsysInterpolateCoordinateAndPower
End Sub

Private Sub CommandButton2_Click()
' Call the backup routine  writting all the parameters before exiting
 backupFormParameters
 Unload Me
End
End Sub

Private Sub reCalculateTotalPowerButton_Click()
    Call writeAnsysHeatFluxElements
End Sub


Private Sub viewFileButton_Click()
  Call OpenFileExplorerAndHighlightFile
End Sub

Sub OpenFileExplorerAndHighlightFile()
 Dim shellCmd As String
 shellCmd = "explorer.exe /select, """ & externalFilePath & """"
 Shell shellCmd, vbNormalFocus

End Sub

Private Sub createExternalFileButton_Click()
 ' write out to a external file
   Call writeAnsysHeatFluxElements
End Sub

Private Sub CheckCreateExternalFileButton()
    If Me.outputFileName.Value <> "" And Me.outputFilePath.Value <> "" Then
        Me.createExternalFileButton.Enabled = True
    Else
        Me.createExternalFileButton.Enabled = False
    End If
End Sub


Private Sub outFilePathButton_Click()
    Dim folderPath As String
    folderPath = ""
    With Application.fileDialog(msoFileDialogFolderPicker)
        .Title = "Select a folder"
        .Show
        If .SelectedItems.Count > 0 Then
            folderPath = .SelectedItems(1)
            ' Check if folder exists
            If Dir(folderPath, vbDirectory) = "" Then
                'Folder doesn't exist, ask user if they want to create it
                If MsgBox("The folder '" & folderPath & "' does not exist. Do you want to create it?", vbYesNo) = vbYes Then
                    MkDir folderPath
                    MsgBox folderPath & " folder created:"
                Else
                    folderPath = ""
                End If
            End If
        Else
            MsgBox "No folder selected."
        End If
    End With
    
    If folderPath <> "" Then
        form1.outputFilePath.Value = folderPath
    End If
    
    ' After setting the outputFilePath, re-check button status
    Call CheckCreateExternalFileButton
End Sub


Private Sub outputFileName_Change()
    Call CheckCreateExternalFileButton
End Sub


Private Sub uploadAnsysFileButton_Click()
    Dim fileDialog As fileDialog
    Set fileDialog = Application.fileDialog(msoFileDialogFilePicker)
    
    With fileDialog
        .Title = "Select a ANSYS APDL file"
        .Filters.Clear
        .Filters.Add "Data file, ", "*.dat"
        .AllowMultiSelect = False
        
        If .Show = True Then
            'Get the selected file path
            Dim FilePath As String
            FilePath = .SelectedItems(1)
            'MsgBox "Selected file: " & filePath
            form1.ansysFileName = FilePath
        End If
    End With
    
' by default using same ansys input filename as inp filename
    Call ansysFileName_AfterUpdate

' read ansys file and store the data
   Call readAnsysFile
' update form information
 With form1
    .ansysTotalNode = Format(ansysNodes.Count, "#,##0")
    If ansysElements.Count > 0 Then
        .ansysTotalElement = Format(ansysElements.Count, "#,##0")
     Else
        .ansysTotalElement = Format(totalElementCount, "#,##0")
    End If
    .totalFluxElement = Format(ansysHeatFluxElements.Count, "#,##0")
 ' make spectra file button enabled
    .uploadSpectraFileButton.Enabled = True
 End With

End Sub

Private Sub uploadSpectraFileButton_Click()
    Dim fileDialog As fileDialog
    Set fileDialog = Application.fileDialog(msoFileDialogFilePicker)
    
    With fileDialog
        .Title = "Select a SPECTRA x-y mesh file"
        .Filters.Clear
        .Filters.Add "Spectra file, ", "*.dta,*.data,*.dta2"
        .AllowMultiSelect = False
        
        If .Show = True Then
            'Get the selected file path
            Dim FilePath As String
            FilePath = .SelectedItems(1)
            'MsgBox "Selected file: " & filePath
            form1.spectraFileName = FilePath
        End If
    End With
' read spectra power density file
    Call readSpectraFile
' determine the centroid point of a heatflux surface to ansysHFluxElement.elementNodes
    Call storeAnsysInterpolateCoordinateAndPower
    
 With form1
      .spectraTotalColumn1.Caption = spectraTotalColumn
      .spectraTotalRow1.Caption = spectraTotalRow
      .spectraMinX.Caption = Format(minX, "0.000") & " mrad"
      .spectraMaxX.Caption = Format(maxX, "0.000") & " mrad"
      .spectraMinY.Caption = Format(minY, "0.000") & " mrad"
      .spectraMaxY.Caption = Format(maxY, "0.000") & " mrad"
      .spectraPeakPowerDensity1.Caption = Format(spectraPeakPowerDensity, "0.00") & " kW/mrad2"
      .spectraTotalPower1.Caption = Format(spectraTotalPower, "0.00") & " kW"
 End With
End Sub

Private Sub ansysFileName_AfterUpdate()
    Dim originalFileName As String
    Dim fileNameOnly As String
    Dim newFileName As String

    ' Get the full path from ansysFileName input
    originalFileName = Me.ansysFileName.Value

    ' Extract just the file name from the path
    fileNameOnly = Mid(originalFileName, InStrRev(originalFileName, "\") + 1)

    ' Check if it has an extension
    If InStrRev(fileNameOnly, ".") > 0 Then
        ' Replace the extension with ".inp"
        newFileName = Left(fileNameOnly, InStrRev(fileNameOnly, ".") - 1) & ".inp"
    Else
        ' If no extension, simply append ".inp"
        newFileName = fileNameOnly & ".inp"
    End If

    ' Set the outputFileName input box to the modified file name
    Me.outputFileName.Value = newFileName
End Sub


Private Sub xSide_Change()

End Sub

Private Sub xSource_Change()

End Sub

Private Sub zSource_Change()

End Sub