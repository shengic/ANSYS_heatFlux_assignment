Option Explicit

' This module provides a utility to save all form parameters to a text file.
' Modify the parameters as per your form controls.

Public Sub backupFormParameters()
    Dim backupPath As String
    Dim ff As Integer
    Dim fName As String
    Dim ansysDir As String
    
    ' Get the ansysFileName from the form
    fName = form1.ansysFileName
    
    ' Extract the directory from the ansysFileName
    If InStrRev(fName, "\") > 0 Then
        ansysDir = Left(fName, InStrRev(fName, "\") - 1)
        fName = Mid(fName, InStrRev(fName, "\") + 1)
    Else
        ' If there's no directory in ansysFileName, use current workbook path as fallback
        ansysDir = ThisWorkbook.Path
    End If
    
    ' Remove the extension if it exists
    If InStrRev(fName, ".") > 0 Then
        fName = Left(fName, InStrRev(fName, ".") - 1)
    End If
    
    ' Construct the backup path using ansysFileName directory as the base
    backupPath = ansysDir & "\" & fName & ".backup.txt"
    
    ff = FreeFile()
    On Error GoTo HandleError
    Open backupPath For Output As #ff
    
    ' Write out the current date/time at the beginning of the file
    Print #ff, "Backup Date: " & Format(Now, "yyyy-mm-dd hh:nn:ss")
    Print #ff, "-----------------------------------------"
    
    ' Write out parameters line by line.
    Print #ff, "----------------ANSYS----------------"
    Print #ff, " "
    Print #ff, "ansysFileName=" & form1.ansysFileName
    Print #ff, "ansysTotalNode=" & form1.ansysTotalNode
    Print #ff, "ansysTotalElement=" & form1.ansysTotalElement
    Print #ff, "totalFluxElement=" & form1.totalFluxElement
    
    Print #ff, "----------------Spectra----------------"
    Print #ff, " "
    Print #ff, "spectraFileName=" & form1.spectraFileName
    Print #ff, "spectraPeakPowerDensity1=" & form1.spectraPeakPowerDensity1
    Print #ff, "spectraTotalPower1=" & form1.spectraTotalPower1
    Print #ff, "spectraMinX=" & form1.spectraMinX
    Print #ff, "spectraMaxX=" & form1.spectraMaxX
    Print #ff, "spectraMinY=" & form1.spectraMinY
    Print #ff, "spectraMaxY=" & form1.spectraMaxY
    
    Print #ff, "spectraTotalColumn1=" & form1.spectraTotalColumn1
    Print #ff, "spectraTotalRow1=" & form1.spectraTotalRow1

    Print #ff, "----------------Geometry----------------"
    Print #ff, " "
    Print #ff, "xSource=" & form1.xSource.Value
    Print #ff, "ySource=" & form1.ySource.Value
    Print #ff, "zSource=" & form1.zSource.Value
    
    Print #ff, "xTarget=" & form1.xTarget.Value
    Print #ff, "yTarget=" & form1.yTarget.Value
    Print #ff, "zTarget=" & form1.zTarget.Value
    
    Print #ff, "xSide=" & form1.xSide.Value
    Print #ff, "ySide=" & form1.ySide.Value
    Print #ff, "zSide=" & form1.zSide.Value

    Print #ff, "outputFilePath=" & form1.outputFilePath.Value
    Print #ff, "outputFileName=" & form1.outputFileName.Value
    Print #ff, "outputFileName=" & form1.outputTotalPower
    
    Print #ff, "totalPowerRatio=" & form1.totalPowerRatio.Value

    Close #ff
    
    MsgBox " Parameters have been backed up to " & backupPath, vbInformation
    
    Exit Sub

HandleError:
    ' In case of an error, ensure the file is closed and report the error.
    If ff <> 0 Then Close #ff
    MsgBox "Error backing up parameters: " & Err.Description, vbCritical

End Sub